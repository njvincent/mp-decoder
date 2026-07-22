#!/bin/bash
#SBATCH --job-name=baseline_time_scan_threaded
#SBATCH --partition=caslake
#SBATCH --account=pi-liangjiang
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/baseline_time_scan_threaded_%A_%a.out
#SBATCH --error=logs/baseline_time_scan_threaded_%A_%a.err

set -euo pipefail

MODE=${1:-Ft}
MAX_CONCURRENT=${2:-30}
JULIA_SCRIPT=${JULIA_SCRIPT:-2d_windowed_baseline.jl}
THREADS_PER_TASK=${THREADS_PER_TASK:-${SLURM_CPUS_PER_TASK:-16}}
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
PROJECT_DIR=${PROJECT_DIR:-${SUBMIT_DIR}}
OUTPUT_DIR=${OUTPUT_DIR:-results/baseline/ft/time_scan}

if [[ "${PROJECT_DIR}" != /* ]]; then
    PROJECT_DIR="${SUBMIT_DIR}/${PROJECT_DIR}"
fi

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi

if [[ "${JULIA_SCRIPT}" != /* ]]; then
    JULIA_SCRIPT="${PROJECT_DIR}/${JULIA_SCRIPT}"
fi

STOP_MODE=${STOP_MODE:-trials}

if [[ "${STOP_MODE}" == "trials" ]]; then
    MAX_TRIALS=${MAX_TRIALS:-100000}
    unset ACC_ERRORS
elif [[ "${STOP_MODE}" == "failures" ]]; then
    ACC_ERRORS=${ACC_ERRORS:-1000}
    unset MAX_TRIALS
else
    echo "STOP_MODE must be either failures or trials."
    exit 1
fi

UPDATE_TIMES=(${UPDATE_TIME_LIST:-1 2 3})
CLEANUP_TIMES=(${CLEANUP_TIME_LIST:-2 4})
BUFFER_DEPTHS=(${BUFFER_DEPTH_LIST:-1})

P_VALUES=(${P_LIST:-0.010 0.011 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019 0.020})
L_VALUES=(${L_LIST:-5 7 9 13 19})
QRAT=${QRAT:-1}
RVAL=${RVAL:-3}
SYNCH=${SYNCH:-true}
LOGZ=${LOGZ:-true}
TRIAL_PARALLEL=${TRIAL_PARALLEL:-true}
JULIA_MODULE=${JULIA_MODULE:-julia}

N_UPDATE_TIME=${#UPDATE_TIMES[@]}
N_CLEANUP_TIME=${#CLEANUP_TIMES[@]}
N_BUFFER_DEPTH=${#BUFFER_DEPTHS[@]}
NP=${#P_VALUES[@]}
NL=${#L_VALUES[@]}
NCOMBOS=$((N_UPDATE_TIME * N_CLEANUP_TIME * N_BUFFER_DEPTH * NP * NL))
TOTAL_TASKS=${NCOMBOS}

if (( N_UPDATE_TIME < 1 || N_CLEANUP_TIME < 1 || N_BUFFER_DEPTH < 1 || NP < 1 || NL < 1 )); then
    echo "UPDATE_TIME_LIST, CLEANUP_TIME_LIST, BUFFER_DEPTH_LIST, P_LIST, and L_LIST must be nonempty."
    exit 1
fi

if (( TOTAL_TASKS < 1 )); then
    echo "No tasks to submit."
    exit 1
fi

if (( MAX_CONCURRENT < 1 )); then
    echo "MAX_CONCURRENT must be at least 1."
    exit 1
fi

if [[ ! -f "${JULIA_SCRIPT}" ]]; then
    echo "Julia script not found: ${JULIA_SCRIPT}"
    exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    export MODE
    export STOP_MODE
    export MAX_CONCURRENT
    export UPDATE_TIME_LIST="${UPDATE_TIMES[*]}"
    export CLEANUP_TIME_LIST="${CLEANUP_TIMES[*]}"
    export BUFFER_DEPTH_LIST="${BUFFER_DEPTHS[*]}"
    export P_LIST="${P_VALUES[*]}"
    export L_LIST="${L_VALUES[*]}"
    export QRAT
    export RVAL
    export SYNCH
    export LOGZ
    export TRIAL_PARALLEL
    if [[ "${STOP_MODE}" == "trials" ]]; then
        export MAX_TRIALS
    else
        export ACC_ERRORS
    fi
    export JULIA_SCRIPT
    export THREADS_PER_TASK
    export JULIA_MODULE
    export PROJECT_DIR
    export OUTPUT_DIR

    mkdir -p logs
    mkdir -p "${OUTPUT_DIR}"

    echo "Submitting ${TOTAL_TASKS} baseline tasks: mode=${MODE}, threads/task=${THREADS_PER_TASK}"
    echo "L=(${L_VALUES[*]}), p=(${P_VALUES[*]})"
    echo "UPDATE_TIME factors=(${UPDATE_TIMES[*]}), CLEANUP_TIME factors=(${CLEANUP_TIMES[*]})"
    echo "BUFFER_DEPTH factors=(${BUFFER_DEPTHS[*]})"
    if [[ "${STOP_MODE}" == "trials" ]]; then
        echo "STOP_MODE=${STOP_MODE}, MAX_TRIALS=${MAX_TRIALS}"
    else
        echo "STOP_MODE=${STOP_MODE}, ACC_ERRORS=${ACC_ERRORS}"
    fi
    echo "max concurrent array jobs=${MAX_CONCURRENT}"
    echo "output dir=${OUTPUT_DIR}"
    sbatch \
        --cpus-per-task="${THREADS_PER_TASK}" \
        --export=ALL \
        --array="0-$((TOTAL_TASKS - 1))%${MAX_CONCURRENT}" \
        "$0" "${MODE}" "${MAX_CONCURRENT}"
    exit 0
fi

module load "${JULIA_MODULE}"

mkdir -p logs

if (( SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} is outside TOTAL_TASKS=${TOTAL_TASKS}; exiting."
    exit 0
fi

COMBO_INDEX=$((SLURM_ARRAY_TASK_ID % NCOMBOS))
TASK_OUTPUT_DIR="${OUTPUT_DIR}"
mkdir -p "${TASK_OUTPUT_DIR}"
INDEX=${COMBO_INDEX}
P_INDEX=$((INDEX % NP))
INDEX=$((INDEX / NP))
L_INDEX=$((INDEX % NL))
INDEX=$((INDEX / NL))
BUFFER_DEPTH_INDEX=$((INDEX % N_BUFFER_DEPTH))
INDEX=$((INDEX / N_BUFFER_DEPTH))
UPDATE_TIME_INDEX=$((INDEX % N_UPDATE_TIME))
INDEX=$((INDEX / N_UPDATE_TIME))
CLEANUP_TIME_INDEX=$((INDEX % N_CLEANUP_TIME))

PVAL=${P_VALUES[$P_INDEX]}
LVAL=${L_VALUES[$L_INDEX]}
BUFFER_DEPTH=${BUFFER_DEPTHS[$BUFFER_DEPTH_INDEX]}
UPDATE_TIME=${UPDATE_TIMES[$UPDATE_TIME_INDEX]}
CLEANUP_TIME=${CLEANUP_TIMES[$CLEANUP_TIME_INDEX]}

export PVAL
export LVAL
export MODE
export STOP_MODE
export BUFFER_DEPTH
export UPDATE_TIME
export CLEANUP_TIME
export QRAT
export RVAL
export SYNCH
export LOGZ
export TRIAL_PARALLEL
if [[ "${STOP_MODE}" == "trials" ]]; then
    export MAX_TRIALS
else
    export ACC_ERRORS
fi
export JULIA_NUM_THREADS="${THREADS_PER_TASK}"
export OUT_ADJ="_thr${JULIA_NUM_THREADS}"

echo "MODE=${MODE} L=${LVAL} p=${PVAL} BUFFER_DEPTH=${BUFFER_DEPTH}×log1.5(L) UPDATE_TIME=${UPDATE_TIME}L CLEANUP_TIME=${CLEANUP_TIME}L"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "JULIA_SCRIPT=${JULIA_SCRIPT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TASK_OUTPUT_DIR=${TASK_OUTPUT_DIR}"
if [[ "${STOP_MODE}" == "trials" ]]; then
    echo "JULIA_NUM_THREADS=${JULIA_NUM_THREADS} TRIAL_PARALLEL=${TRIAL_PARALLEL} STOP_MODE=${STOP_MODE} MAX_TRIALS=${MAX_TRIALS}"
else
    echo "JULIA_NUM_THREADS=${JULIA_NUM_THREADS} TRIAL_PARALLEL=${TRIAL_PARALLEL} STOP_MODE=${STOP_MODE} ACC_ERRORS=${ACC_ERRORS}"
fi

(
    cd "${TASK_OUTPUT_DIR}"
    julia --threads="${JULIA_NUM_THREADS}" "${JULIA_SCRIPT}"
)
