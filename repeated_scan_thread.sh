#!/bin/bash
#SBATCH --job-name=mp_decoder_repeats_threaded
#SBATCH --partition=caslake
#SBATCH --account=pi-liangjiang
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/mp_decoder_repeats_threaded_%A_%a.out
#SBATCH --error=logs/mp_decoder_repeats_threaded_%A_%a.err

set -euo pipefail

MODE=${1:-Ft}
NREPEATS=${2:-1}
MAX_CONCURRENT=${3:-15}
JULIA_SCRIPT=${JULIA_SCRIPT:-2d_windowed_simulation_thread.jl}
THREADS_PER_TASK=${THREADS_PER_TASK:-${SLURM_CPUS_PER_TASK:-16}}

P_LIST=(${P_LIST:-0.011 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019})
L_LIST=(${L_LIST:-5 7 9 13 19})
QRAT=${QRAT:-1}
RVAL=${RVAL:-3}
SYNCH=${SYNCH:-true}
LOGZ=${LOGZ:-true}
TRIAL_PARALLEL=${TRIAL_PARALLEL:-true}
JULIA_MODULE=${JULIA_MODULE:-julia}

NP=${#P_LIST[@]}
NL=${#L_LIST[@]}
NCOMBOS=$((NP * NL))
TOTAL_TASKS=$((NP * NL * NREPEATS))

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
    export QRAT
    export RVAL
    export SYNCH
    export LOGZ
    export TRIAL_PARALLEL
    export JULIA_SCRIPT
    export THREADS_PER_TASK
    export JULIA_MODULE

    mkdir -p logs

    echo "Submitting ${TOTAL_TASKS} RCC tasks: mode=${MODE}, repeats=${NREPEATS}, threads/task=${THREADS_PER_TASK}"
    echo "L=(${L_LIST[*]}), p=(${P_LIST[*]}), max concurrent array jobs=${MAX_CONCURRENT}"
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
L_INDEX=$((COMBO_INDEX / NP))

PVAL=${P_LIST[$P_INDEX]}
LVAL=${L_LIST[$L_INDEX]}

export PVAL
export LVAL
export MODE
export QRAT
export RVAL
export SYNCH
export LOGZ
export REPEAT_INDEX
export TRIAL_PARALLEL
export JULIA_NUM_THREADS="${THREADS_PER_TASK}"
export OUT_ADJ="_p${PVAL}_L${LVAL}_rep${REPEAT_INDEX}_thr${JULIA_NUM_THREADS}"

echo "MODE=${MODE} L=${LVAL} p=${PVAL} repeat=${REPEAT_INDEX}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "JULIA_SCRIPT=${JULIA_SCRIPT}"
echo "JULIA_NUM_THREADS=${JULIA_NUM_THREADS} TRIAL_PARALLEL=${TRIAL_PARALLEL}"

julia --threads="${JULIA_NUM_THREADS}" "${JULIA_SCRIPT}"
