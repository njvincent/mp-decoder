#!/bin/bash
#SBATCH --job-name=cnot_consecutive_count
#SBATCH --partition=caslake
#SBATCH --account=pi-liangjiang
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/cnot_consecutive_count_%A_%a.out
#SBATCH --error=logs/cnot_consecutive_count_%A_%a.err

set -euo pipefail

# Each k-CNOT sequence uses k + 1 interval factors, all equal to 1:
#   interval 1, C1 -> target, ..., interval 1, Ck -> target, interval 1.
# As in cnot_primitive_consecutive_full.sh, factor 1 resolves to TVAL noisy rounds.

# Set the largest consecutive-CNOT count to scan here. The scan includes every
# count from 1 through MAX_CNOTS.
MAX_CNOTS=5

MODE=${1:-CNOT_Ft}
NREPEATS=${2:-1}
MAX_CONCURRENT=${3:-30}

if [[ ! "${MAX_CNOTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_CNOTS must be a positive integer." >&2
    exit 1
fi
if [[ ! "${NREPEATS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NREPEATS must be a positive integer." >&2
    exit 1
fi
if [[ ! "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_CONCURRENT must be a positive integer." >&2
    exit 1
fi

JULIA_SCRIPT=${JULIA_SCRIPT:-cnot_primitive_consecutive.jl}
THREADS_PER_TASK=${THREADS_PER_TASK:-${SLURM_CPUS_PER_TASK:-16}}
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
PROJECT_DIR=${PROJECT_DIR:-${SUBMIT_DIR}}
OUTPUT_DIR=${OUTPUT_DIR:-results/cnot_primitive_consecutive/count_scan_interval_1}

if [[ "${PROJECT_DIR}" != /* ]]; then
    PROJECT_DIR="${SUBMIT_DIR}/${PROJECT_DIR}"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi
if [[ "${JULIA_SCRIPT}" != /* ]]; then
    JULIA_SCRIPT="${PROJECT_DIR}/${JULIA_SCRIPT}"
fi

P_LIST=(${P_LIST:-0.006 0.007 0.008 0.009 0.010 0.011 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019})
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

# Select either a fixed-trial or failure target.
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

NP=${#P_LIST[@]}
NL=${#L_LIST[@]}
NBASE_COMBOS=$((NP * NL))
NCOMBOS_PER_REPEAT=$((NBASE_COMBOS * MAX_CNOTS))
TOTAL_TASKS=$((NCOMBOS_PER_REPEAT * NREPEATS))

if (( TOTAL_TASKS < 1 )); then
    echo "No tasks to submit." >&2
    exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    export MAX_CNOTS
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

    echo "Submitting ${TOTAL_TASKS} consecutive primitive-CNOT count-scan tasks"
    echo "CNOT counts=1..${MAX_CNOTS}, interval factor=1"
    echo "mode=${MODE}, repeats=${NREPEATS}, threads/task=${THREADS_PER_TASK}, max concurrent=${MAX_CONCURRENT}"
    echo "L=(${L_LIST[*]}), p=(${P_LIST[*]}), qrat=${QRAT}, r=${RVAL}, synch=${SYNCH}, logZ=${LOGZ}"
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

if type module >/dev/null 2>&1; then
    module load "${JULIA_MODULE}"
else
    echo "Environment modules unavailable; using Julia already on PATH."
fi

if (( SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside TOTAL_TASKS=${TOTAL_TASKS}; exiting."
    exit 0
fi

CNOT_INDEX=$((SLURM_ARRAY_TASK_ID % MAX_CNOTS))
BASE_COMBO_INDEX=$(((SLURM_ARRAY_TASK_ID / MAX_CNOTS) % NBASE_COMBOS))
REPEAT_INDEX=$((SLURM_ARRAY_TASK_ID / NCOMBOS_PER_REPEAT))
P_INDEX=$((BASE_COMBO_INDEX % NP))
L_INDEX=$((BASE_COMBO_INDEX / NP))

CNOT_COUNT=$((CNOT_INDEX + 1))
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

# A sequence with k CNOTs requires k + 1 noisy intervals. Interval factor 1
# means every resolved interval is exactly TVAL rounds.
RESOLVED_INTERVALS=()
for ((interval_index = 0; interval_index <= CNOT_COUNT; interval_index++)); do
    RESOLVED_INTERVALS+=("${TVAL_VAL}")
done

printf -v CNOT_INTERVALS_RESOLVED '%s,' "${RESOLVED_INTERVALS[@]}"
CNOT_INTERVALS_RESOLVED=${CNOT_INTERVALS_RESOLVED%,}

if [[ "${CLEANUP_TIME_DEFAULT}" == "auto" ]]; then
    CLEANUP_TIME_VAL=$((2 * TVAL_VAL))
else
    CLEANUP_TIME_VAL=${CLEANUP_TIME_DEFAULT}
fi
if [[ ! "${CLEANUP_TIME_VAL}" =~ ^[0-9]+$ ]]; then
    echo "CLEANUP_TIME must resolve to a nonnegative integer." >&2
    exit 1
fi

printf -v CNOT_DIR "cnot_count_%03d" "${CNOT_COUNT}"
printf -v REPEAT_DIR "rep%02d" "${REPEAT_INDEX}"
TASK_OUTPUT_DIR="${OUTPUT_DIR}/${CNOT_DIR}/${REPEAT_DIR}"
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
export OUT_ADJ="_consecutive_count_scan_k${CNOT_COUNT}_rep${REPEAT_INDEX}_T${TVAL_VAL}_thr${JULIA_NUM_THREADS}"

echo "MODE=${MODE} L=${LVAL} p=${PVAL} repeat=${REPEAT_INDEX}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "JULIA_SCRIPT=${JULIA_SCRIPT}"
echo "TASK_OUTPUT_DIR=${TASK_OUTPUT_DIR}"
echo "JULIA_NUM_THREADS=${JULIA_NUM_THREADS} TRIAL_PARALLEL=${TRIAL_PARALLEL}"
echo "CNOT_STYLE=${CNOT_STYLE} TVAL=${TVAL} CNOT_COUNT=${CNOT_COUNT}"
echo "interval factor=1, resolved CNOT_INTERVALS=${CNOT_INTERVALS}"
echo "CLEANUP_TIME=${CLEANUP_TIME}"
if [[ "${STOP_MODE}" == "trials" ]]; then
    echo "STOP_MODE=${STOP_MODE}, MAX_TRIALS=${MAX_TRIALS}"
else
    echo "STOP_MODE=${STOP_MODE}, ACC_ERRORS=${ACC_ERRORS}"
fi

(
    cd "${TASK_OUTPUT_DIR}"
    julia --threads="${JULIA_NUM_THREADS}" "${JULIA_SCRIPT}"
)
