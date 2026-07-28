#!/bin/bash
#SBATCH --job-name=cnot_consecutive_initial
#SBATCH --partition=caslake
#SBATCH --account=pi-liangjiang
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/cnot_consecutive_initial_%A_%a.out
#SBATCH --error=logs/cnot_consecutive_initial_%A_%a.err

set -euo pipefail

MODE=${1:-CNOT_Ft}
NREPEATS=${2:-1}
MAX_CONCURRENT=${3:-20}
JULIA_SCRIPT=${JULIA_SCRIPT:-cnot_primitive_consecutive.jl}
THREADS_PER_TASK=${THREADS_PER_TASK:-${SLURM_CPUS_PER_TASK:-16}}
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
PROJECT_DIR=${PROJECT_DIR:-${SUBMIT_DIR}}
OUTPUT_DIR=${OUTPUT_DIR:-results/cnot_primitive_consecutive/initial_scan}

if [[ "${PROJECT_DIR}" != /* ]]; then
    PROJECT_DIR="${SUBMIT_DIR}/${PROJECT_DIR}"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi
if [[ "${JULIA_SCRIPT}" != /* ]]; then
    JULIA_SCRIPT="${PROJECT_DIR}/${JULIA_SCRIPT}"
fi

P_LIST=(${P_LIST:-0.004 0.006 0.008 0.010 0.012 0.014 0.016 0.018})
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

# Consecutive noisy intervals, in multiples of TVAL. For example, (1 1 1)
# means T rounds, C1 -> target, T rounds, C2 -> target, then T rounds.
# Entries may be nonnegative integers or fractions such as 1/2.
CNOT_INTERVAL_FACTORS=(1 1 1)

if (( ${#CNOT_INTERVAL_FACTORS[@]} < 2 )); then
    echo "A CNOT sequence requires at least two interval entries." >&2
    exit 1
fi

parse_nonnegative_factor() {
    local spec=$1
    if [[ "${spec}" =~ ^([0-9]+)$ ]]; then
        FACTOR_NUM=${BASH_REMATCH[1]}
        FACTOR_DEN=1
    elif [[ "${spec}" =~ ^([0-9]+)/([1-9][0-9]*)$ ]]; then
        FACTOR_NUM=${BASH_REMATCH[1]}
        FACTOR_DEN=${BASH_REMATCH[2]}
    else
        echo "Invalid interval factor '${spec}'; use n or n/d." >&2
        exit 1
    fi
}

for interval_factor in "${CNOT_INTERVAL_FACTORS[@]}"; do
    parse_nonnegative_factor "${interval_factor}"
done

SCHEDULE_LABEL=factors
for interval_factor in "${CNOT_INTERVAL_FACTORS[@]}"; do
    interval_label=${interval_factor//\//div}
    SCHEDULE_LABEL+="-${interval_label}"
done

# Select either a fixed-trial or failure target.
STOP_MODE=${STOP_MODE:-trials}
if [[ "${STOP_MODE}" == "trials" ]]; then
    MAX_TRIALS=${MAX_TRIALS:-1000}
    unset ACC_ERRORS
elif [[ "${STOP_MODE}" == "failures" ]]; then
    ACC_ERRORS=${ACC_ERRORS:-1000}
    unset MAX_TRIALS
else
    echo "STOP_MODE must be either failures or trials." >&2
    exit 1
fi
unset SAMPS

NP=${#P_LIST[@]}
NL=${#L_LIST[@]}
NCOMBOS=$((NP * NL))
TOTAL_TASKS=$((NCOMBOS * NREPEATS))

if (( TOTAL_TASKS < 1 )); then
    echo "No tasks to submit." >&2
    exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    export MODE
    export NREPEATS
    export MAX_CONCURRENT
    export P_LIST="${P_LIST[*]}"
    export L_LIST="${L_LIST[*]}"
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

    echo "Submitting ${TOTAL_TASKS} consecutive primitive-CNOT tasks"
    echo "mode=${MODE}, repeats=${NREPEATS}, threads/task=${THREADS_PER_TASK}, max concurrent=${MAX_CONCURRENT}"
    echo "L=(${L_LIST[*]}), p=(${P_LIST[*]}), qrat=${QRAT}, r=${RVAL}, synch=${SYNCH}, logZ=${LOGZ}"
    echo "interval factors=(${CNOT_INTERVAL_FACTORS[*]})"
    echo "TVAL=${TVAL_DEFAULT}, CLEANUP_TIME=${CLEANUP_TIME_DEFAULT}"
    if [[ "${STOP_MODE}" == "trials" ]]; then
        echo "STOP_MODE=${STOP_MODE}, MAX_TRIALS=${MAX_TRIALS}"
    else
        echo "STOP_MODE=${STOP_MODE}, ACC_ERRORS=${ACC_ERRORS}"
    fi
    echo "output dir=${OUTPUT_DIR}/${SCHEDULE_LABEL}"

    sbatch \
        --cpus-per-task="${THREADS_PER_TASK}" \
        --export=ALL \
        --array="0-$((TOTAL_TASKS - 1))%${MAX_CONCURRENT}" \
        "$0" "${MODE}" "${NREPEATS}" "${MAX_CONCURRENT}"
    exit 0
fi

if type module >/dev/null 2>&1; then
    module load "${JULIA_MODULE}"
else
    echo "Environment modules unavailable; using Julia already on PATH."
fi

if (( SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside TOTAL_TASKS=${TOTAL_TASKS}; exiting."
    exit 0
fi

COMBO_INDEX=$((SLURM_ARRAY_TASK_ID % NCOMBOS))
REPEAT_INDEX=$((SLURM_ARRAY_TASK_ID / NCOMBOS))
P_INDEX=$((COMBO_INDEX % NP))
L_INDEX=$((COMBO_INDEX / NP))

PVAL=${P_LIST[$P_INDEX]}
LVAL=${L_LIST[$L_INDEX]}
if [[ "${TVAL_DEFAULT}" == "L" ]]; then
    TVAL_VAL=${LVAL}
else
    TVAL_VAL=${TVAL_DEFAULT}
fi
if [[ ! "${TVAL_VAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Resolved TVAL must be a positive integer; got '${TVAL_VAL}'." >&2
    exit 1
fi

RESOLVED_INTERVALS=()
for interval_factor in "${CNOT_INTERVAL_FACTORS[@]}"; do
    parse_nonnegative_factor "${interval_factor}"
    resolved_interval=$((TVAL_VAL * 10#${FACTOR_NUM} / 10#${FACTOR_DEN}))
    RESOLVED_INTERVALS+=("${resolved_interval}")
done

printf -v CNOT_INTERVALS_RESOLVED '%s,' "${RESOLVED_INTERVALS[@]}"
CNOT_INTERVALS_RESOLVED=${CNOT_INTERVALS_RESOLVED%,}
CNOT_COUNT=$((${#RESOLVED_INTERVALS[@]} - 1))

if [[ "${CLEANUP_TIME_DEFAULT}" == "auto" ]]; then
    CLEANUP_TIME_VAL=$((2 * TVAL_VAL))
else
    CLEANUP_TIME_VAL=${CLEANUP_TIME_DEFAULT}
fi
if [[ ! "${CLEANUP_TIME_VAL}" =~ ^[0-9]+$ ]]; then
    echo "CLEANUP_TIME must resolve to a nonnegative integer." >&2
    exit 1
fi

printf -v REPEAT_DIR "rep%02d" "${REPEAT_INDEX}"
TASK_OUTPUT_DIR="${OUTPUT_DIR}/${SCHEDULE_LABEL}/${REPEAT_DIR}"
mkdir -p "${TASK_OUTPUT_DIR}"

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
export CNOT_INTERVALS="${CNOT_INTERVALS_RESOLVED}"
export CLEANUP_TIME="${CLEANUP_TIME_VAL}"
export JULIA_NUM_THREADS="${THREADS_PER_TASK}"
export OUT_ADJ="_consecutive_initial_rep${REPEAT_INDEX}_T${TVAL_VAL}_thr${JULIA_NUM_THREADS}"

echo "MODE=${MODE} L=${LVAL} p=${PVAL} repeat=${REPEAT_INDEX}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "JULIA_SCRIPT=${JULIA_SCRIPT}"
echo "TASK_OUTPUT_DIR=${TASK_OUTPUT_DIR}"
echo "JULIA_NUM_THREADS=${JULIA_NUM_THREADS} TRIAL_PARALLEL=${TRIAL_PARALLEL}"
echo "CNOT_STYLE=${CNOT_STYLE} TVAL=${TVAL} CNOT_COUNT=${CNOT_COUNT}"
echo "interval factors=(${CNOT_INTERVAL_FACTORS[*]})"
echo "resolved CNOT_INTERVALS=${CNOT_INTERVALS} CLEANUP_TIME=${CLEANUP_TIME}"
if [[ "${STOP_MODE}" == "trials" ]]; then
    echo "STOP_MODE=${STOP_MODE}, MAX_TRIALS=${MAX_TRIALS}"
else
    echo "STOP_MODE=${STOP_MODE}, ACC_ERRORS=${ACC_ERRORS}"
fi

(
    cd "${TASK_OUTPUT_DIR}"
    julia --threads="${JULIA_NUM_THREADS}" "${JULIA_SCRIPT}"
)
