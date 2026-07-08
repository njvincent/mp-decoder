#!/bin/bash
#SBATCH --job-name=baseline_full_scan_threaded
#SBATCH --partition=caslake
#SBATCH --account=pi-liangjiang
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/baseline_full_scan_threaded_%A_%a.out
#SBATCH --error=logs/baseline_full_scan_threaded_%A_%a.err

set -euo pipefail

MODE=${1:-Ft}
NREPEATS=${2:-5}
MAX_CONCURRENT=${3:-15}
JULIA_SCRIPT=${JULIA_SCRIPT:-2d_windowed_simulation_thread.jl}
THREADS_PER_TASK=${THREADS_PER_TASK:-${SLURM_CPUS_PER_TASK:-16}}
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
PROJECT_DIR=${PROJECT_DIR:-${SUBMIT_DIR}}

if [[ -z "${OUTPUT_DIR:-}" ]]; then
    case "${MODE}" in
        Ft) OUTPUT_DIR="results/baseline/ft/full_scan" ;;
        trel) OUTPUT_DIR="results/baseline/trel/full_scan" ;;
        *) OUTPUT_DIR="results/baseline/${MODE}/full_scan" ;;
    esac
fi

if [[ "${PROJECT_DIR}" != /* ]]; then
    PROJECT_DIR="${SUBMIT_DIR}/${PROJECT_DIR}"
fi

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi

if [[ "${JULIA_SCRIPT}" != /* ]]; then
    JULIA_SCRIPT="${PROJECT_DIR}/${JULIA_SCRIPT}"
fi

P_VALUES=(${P_LIST:-0.011 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019 0.020})
L_VALUES=(${L_LIST:-5 7 9 13 19})
QRAT=${QRAT:-1}
RVAL=${RVAL:-3}
SYNCH=${SYNCH:-true}
LOGZ=${LOGZ:-true}
TRIAL_PARALLEL=${TRIAL_PARALLEL:-true}
ACC_ERRORS=${ACC_ERRORS:-1000}
JULIA_MODULE=${JULIA_MODULE:-julia}

NP=${#P_VALUES[@]}
NL=${#L_VALUES[@]}
NCOMBOS=$((NP * NL))
TOTAL_TASKS=$((NP * NL * NREPEATS))

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
    export NREPEATS
    export MAX_CONCURRENT
    export P_LIST="${P_VALUES[*]}"
    export L_LIST="${L_VALUES[*]}"
    export QRAT
    export RVAL
    export SYNCH
    export LOGZ
    export TRIAL_PARALLEL
    export ACC_ERRORS
    export JULIA_SCRIPT
    export THREADS_PER_TASK
    export JULIA_MODULE
    export PROJECT_DIR
    export OUTPUT_DIR

    mkdir -p logs
    mkdir -p "${OUTPUT_DIR}"

    echo "Submitting ${TOTAL_TASKS} baseline tasks: mode=${MODE}, repeats=${NREPEATS}, threads/task=${THREADS_PER_TASK}"
    echo "L=(${L_VALUES[*]}), p=(${P_VALUES[*]}), max concurrent array jobs=${MAX_CONCURRENT}"
    echo "ACC_ERRORS=${ACC_ERRORS}"
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
printf -v REPEAT_DIR "rep%02d" "${REPEAT_INDEX}"
TASK_OUTPUT_DIR="${OUTPUT_DIR}/${REPEAT_DIR}"
mkdir -p "${TASK_OUTPUT_DIR}"
P_INDEX=$((COMBO_INDEX % NP))
L_INDEX=$((COMBO_INDEX / NP))

PVAL=${P_VALUES[$P_INDEX]}
LVAL=${L_VALUES[$L_INDEX]}

export PVAL
export LVAL
export MODE
export QRAT
export RVAL
export SYNCH
export LOGZ
export REPEAT_INDEX
export TRIAL_PARALLEL
export ACC_ERRORS
export JULIA_NUM_THREADS="${THREADS_PER_TASK}"
export OUT_ADJ="_p${PVAL}_L${LVAL}_rep${REPEAT_INDEX}_acc${ACC_ERRORS}_thr${JULIA_NUM_THREADS}"

echo "MODE=${MODE} L=${LVAL} p=${PVAL} repeat=${REPEAT_INDEX}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "JULIA_SCRIPT=${JULIA_SCRIPT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TASK_OUTPUT_DIR=${TASK_OUTPUT_DIR}"
echo "JULIA_NUM_THREADS=${JULIA_NUM_THREADS} TRIAL_PARALLEL=${TRIAL_PARALLEL} ACC_ERRORS=${ACC_ERRORS}"

(
    cd "${TASK_OUTPUT_DIR}"
    julia --threads="${JULIA_NUM_THREADS}" "${JULIA_SCRIPT}"
)
