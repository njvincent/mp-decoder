#!/bin/bash
#SBATCH --job-name=cnot_split_scan
#SBATCH --partition=caslake
#SBATCH --account=pi-liangjiang
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/cnot_split_scan_%A_%a.out
#SBATCH --error=logs/cnot_split_scan_%A_%a.err

set -euo pipefail

MODE=${1:-CNOT_Ft}
NREPEATS=${2:-1}
MAX_CONCURRENT=${3:-20}
JULIA_SCRIPT=${JULIA_SCRIPT:-2d_windowed_cnot_primitive.jl}
THREADS_PER_TASK=${THREADS_PER_TASK:-${SLURM_CPUS_PER_TASK:-16}}
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
PROJECT_DIR=${PROJECT_DIR:-${SUBMIT_DIR}}
OUTPUT_DIR=${OUTPUT_DIR:-results/cnot_primitive/split_scan}

if [[ "${PROJECT_DIR}" != /* ]]; then
    PROJECT_DIR="${SUBMIT_DIR}/${PROJECT_DIR}"
fi

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi

if [[ "${JULIA_SCRIPT}" != /* ]]; then
    JULIA_SCRIPT="${PROJECT_DIR}/${JULIA_SCRIPT}"
fi

P_LIST=(${P_LIST:-0.010 0.011 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019 0.020})
L_LIST=(${L_LIST:-5 7 9 13 19})
QRAT=${QRAT:-1}
RVAL=${RVAL:-3}
SYNCH=${SYNCH:-true}
LOGZ=${LOGZ:-true}
TRIAL_PARALLEL=${TRIAL_PARALLEL:-true}
CNOT_STYLE=${CNOT_STYLE:-primitive}
TVAL_DEFAULT=${TVAL:-L}
CLEANUP_TIME_DEFAULT=${CLEANUP_TIME:-auto}
JULIA_MODULE=${JULIA_MODULE:-julia}

# Match 2d_windowed_baseline.jl: select either a failure or fixed-trial target.
# Example: STOP_MODE=trials MAX_TRIALS=100000 bash jobs/cnot_primitive_split_scan.sh
STOP_MODE=${STOP_MODE:-trials}
if [[ "${STOP_MODE}" == "trials" ]]; then
    MAX_TRIALS=${MAX_TRIALS:-100000}
    unset ACC_ERRORS
elif [[ "${STOP_MODE}" == "failures" ]]; then
    ACC_ERRORS=${ACC_ERRORS:-1000}
    unset MAX_TRIALS
else
    echo "STOP_MODE must be either failures or trials." >&2
    exit 1
fi
unset SAMPS

# Each entry is PRE_FRACTION:POST_FRACTION and must sum to one. Override with,
# for example, CNOT_SPLITS="1/3:2/3 2/3:1/3".
CNOT_SPLITS_DEFAULT="1:0 0:1 1/2:1/2 1/4:3/4 3/4:1/4"
CNOT_SPLIT_SPECS=(${CNOT_SPLITS:-${CNOT_SPLITS_DEFAULT}})

parse_fraction() {
    local spec=$1
    if [[ "${spec}" =~ ^([0-9]+)$ ]]; then
        FRACTION_NUM=${BASH_REMATCH[1]}
        FRACTION_DEN=1
    elif [[ "${spec}" =~ ^([0-9]+)/([1-9][0-9]*)$ ]]; then
        FRACTION_NUM=${BASH_REMATCH[1]}
        FRACTION_DEN=${BASH_REMATCH[2]}
    else
        echo "Invalid fraction '${spec}'; use a nonnegative integer or n/d." >&2
        exit 1
    fi
    if (( FRACTION_NUM > FRACTION_DEN )); then
        echo "Fraction '${spec}' must be between 0 and 1." >&2
        exit 1
    fi
}

fraction_label() {
    local numerator=$1
    local denominator=$2
    if (( numerator == 0 )); then
        FRACTION_LABEL=0
    elif (( numerator == denominator )); then
        FRACTION_LABEL=T
    elif (( numerator == 1 )); then
        FRACTION_LABEL="Tdiv${denominator}"
    else
        FRACTION_LABEL="${numerator}Tdiv${denominator}"
    fi
}

PRE_SPECS=()
POST_SPECS=()
PRE_NUMS=()
PRE_DENS=()
SPLIT_LABELS=()

for split_spec in "${CNOT_SPLIT_SPECS[@]}"; do
    IFS=: read -r pre_spec post_spec extra <<< "${split_spec}"
    if [[ -z "${pre_spec}" || -z "${post_spec}" || -n "${extra:-}" ]]; then
        echo "Invalid split '${split_spec}'; use PRE_FRACTION:POST_FRACTION." >&2
        exit 1
    fi

    parse_fraction "${pre_spec}"
    pre_num=${FRACTION_NUM}
    pre_den=${FRACTION_DEN}
    parse_fraction "${post_spec}"
    post_num=${FRACTION_NUM}
    post_den=${FRACTION_DEN}

    if (( pre_num * post_den + post_num * pre_den != pre_den * post_den )); then
        echo "Split '${split_spec}' does not sum to 1." >&2
        exit 1
    fi

    fraction_label "${pre_num}" "${pre_den}"
    pre_label=${FRACTION_LABEL}
    fraction_label "${post_num}" "${post_den}"
    post_label=${FRACTION_LABEL}

    PRE_SPECS+=("${pre_spec}")
    POST_SPECS+=("${post_spec}")
    PRE_NUMS+=("${pre_num}")
    PRE_DENS+=("${pre_den}")
    SPLIT_LABELS+=("${pre_label}_CNOT_${post_label}")
done

NP=${#P_LIST[@]}
NL=${#L_LIST[@]}
NSPLITS=${#CNOT_SPLIT_SPECS[@]}
NCOMBOS=$((NP * NL * NSPLITS))
TOTAL_TASKS=$((NCOMBOS * NREPEATS))

if (( TOTAL_TASKS < 1 )); then
    echo "No tasks to submit."
    exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    export MODE
    export NREPEATS
    export MAX_CONCURRENT
    export P_LIST="${P_LIST[*]}"
    export L_LIST="${L_LIST[*]}"
    export CNOT_SPLITS="${CNOT_SPLIT_SPECS[*]}"
    export QRAT
    export RVAL
    export SYNCH
    export LOGZ
    export TRIAL_PARALLEL
    export CNOT_STYLE
    export STOP_MODE
    if [[ "${STOP_MODE}" == "trials" ]]; then
        export MAX_TRIALS
    else
        export ACC_ERRORS
    fi
    export TVAL="${TVAL_DEFAULT}"
    export CLEANUP_TIME="${CLEANUP_TIME_DEFAULT}"
    export JULIA_SCRIPT
    export THREADS_PER_TASK
    export JULIA_MODULE
    export PROJECT_DIR
    export OUTPUT_DIR

    mkdir -p logs
    mkdir -p "${OUTPUT_DIR}"

    echo "Submitting ${TOTAL_TASKS} primitive CNOT split-scan tasks"
    echo "mode=${MODE}, repeats=${NREPEATS}, threads/task=${THREADS_PER_TASK}, max concurrent=${MAX_CONCURRENT}"
    echo "L=(${L_LIST[*]}), p=(${P_LIST[*]}), qrat=${QRAT}, r=${RVAL}, synch=${SYNCH}, logZ=${LOGZ}"
    echo "splits=(${CNOT_SPLIT_SPECS[*]})"
    echo "TVAL=${TVAL_DEFAULT}, CLEANUP_TIME=${CLEANUP_TIME_DEFAULT}"
    if [[ "${STOP_MODE}" == "trials" ]]; then
        echo "STOP_MODE=${STOP_MODE}, MAX_TRIALS=${MAX_TRIALS}"
    else
        echo "STOP_MODE=${STOP_MODE}, ACC_ERRORS=${ACC_ERRORS}"
    fi
    echo "output dir=${OUTPUT_DIR}"
    sbatch \
        --cpus-per-task="${THREADS_PER_TASK}" \
        --export=ALL \
        --array="0-$((TOTAL_TASKS - 1))%${MAX_CONCURRENT}" \
        "$0" "${MODE}" "${NREPEATS}" "${MAX_CONCURRENT}"
    exit 0
fi

module load "${JULIA_MODULE}"

mkdir -p logs

if (( SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside TOTAL_TASKS=${TOTAL_TASKS}; exiting."
    exit 0
fi

COMBO_INDEX=$((SLURM_ARRAY_TASK_ID % NCOMBOS))
REPEAT_INDEX=$((SLURM_ARRAY_TASK_ID / NCOMBOS))
P_INDEX=$((COMBO_INDEX % NP))
L_INDEX=$(((COMBO_INDEX / NP) % NL))
SPLIT_INDEX=$((COMBO_INDEX / (NP * NL)))

printf -v REPEAT_DIR "rep%02d" "${REPEAT_INDEX}"
SPLIT_DIR=${SPLIT_LABELS[$SPLIT_INDEX]}
TASK_OUTPUT_DIR="${OUTPUT_DIR}/${SPLIT_DIR}/${REPEAT_DIR}"
mkdir -p "${TASK_OUTPUT_DIR}"

PVAL=${P_LIST[$P_INDEX]}
LVAL=${L_LIST[$L_INDEX]}
T_PRE_FRACTION=${PRE_SPECS[$SPLIT_INDEX]}
T_POST_FRACTION=${POST_SPECS[$SPLIT_INDEX]}
T_PRE_NUM=${PRE_NUMS[$SPLIT_INDEX]}
T_PRE_DEN=${PRE_DENS[$SPLIT_INDEX]}

if [[ "${TVAL_DEFAULT}" == "L" ]]; then
    TVAL_VAL=${LVAL}
else
    TVAL_VAL=${TVAL_DEFAULT}
fi

T_PRE_VAL=$((TVAL_VAL * T_PRE_NUM / T_PRE_DEN))
T_POST_VAL=$((TVAL_VAL - T_PRE_VAL))

if [[ "${CLEANUP_TIME_DEFAULT}" == "auto" ]]; then
    CLEANUP_TIME_VAL=$((2 * TVAL_VAL))
else
    CLEANUP_TIME_VAL=${CLEANUP_TIME_DEFAULT}
fi

export PVAL
export LVAL
export MODE
export QRAT
export RVAL
export SYNCH
export LOGZ
export REPEAT_INDEX
export TRIAL_PARALLEL
export CNOT_STYLE
export STOP_MODE
if [[ "${STOP_MODE}" == "trials" ]]; then
    export MAX_TRIALS
else
    export ACC_ERRORS
fi
export TVAL="${TVAL_VAL}"
unset CNOT_T_PRE CNOT_T_POST
export CNOT_T_PRE_FRACTION="${T_PRE_FRACTION}"
export CNOT_T_POST_FRACTION="${T_POST_FRACTION}"
export CLEANUP_TIME="${CLEANUP_TIME_VAL}"
export JULIA_NUM_THREADS="${THREADS_PER_TASK}"
export OUT_ADJ="_cnot_split_${SPLIT_DIR}_p${PVAL}_L${LVAL}_rep${REPEAT_INDEX}_T${TVAL_VAL}_Tpre${T_PRE_VAL}_Tpost${T_POST_VAL}_thr${JULIA_NUM_THREADS}"

echo "MODE=${MODE} L=${LVAL} p=${PVAL} split=${SPLIT_DIR} repeat=${REPEAT_INDEX}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "JULIA_SCRIPT=${JULIA_SCRIPT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TASK_OUTPUT_DIR=${TASK_OUTPUT_DIR}"
echo "JULIA_NUM_THREADS=${JULIA_NUM_THREADS} TRIAL_PARALLEL=${TRIAL_PARALLEL}"
echo "CNOT_STYLE=${CNOT_STYLE} TVAL=${TVAL} PRE_FRACTION=${CNOT_T_PRE_FRACTION} POST_FRACTION=${CNOT_T_POST_FRACTION}"
echo "resolved T_PRE=${T_PRE_VAL} T_POST=${T_POST_VAL} CLEANUP_TIME=${CLEANUP_TIME}"
if [[ "${STOP_MODE}" == "trials" ]]; then
    echo "STOP_MODE=${STOP_MODE}, MAX_TRIALS=${MAX_TRIALS}"
else
    echo "STOP_MODE=${STOP_MODE}, ACC_ERRORS=${ACC_ERRORS}"
fi

(
    cd "${TASK_OUTPUT_DIR}"
    julia --threads="${JULIA_NUM_THREADS}" "${JULIA_SCRIPT}"
)
