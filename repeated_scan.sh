#!/bin/bash
#SBATCH --job-name=mp_decoder_repeats
#SBATCH --partition=caslake
#SBATCH --account=pi-liangjiang
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/mp_decoder_repeats_%A_%a.out
#SBATCH --error=logs/mp_decoder_repeats_%A_%a.err

set -euo pipefail

MODE=${1:-Ft}
NREPEATS=${2:-1}
MAX_CONCURRENT=${3:-15}
JULIA_SCRIPT=${JULIA_SCRIPT:-2d_windowed_simulation.jl}

P_LIST=(${P_LIST:-0.011 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019})
L_LIST=(${L_LIST:-5 7 9 13 19})
QRAT=${QRAT:-1}
RVAL=${RVAL:-3}
SYNCH=${SYNCH:-true}
LOGZ=${LOGZ:-true}

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
    export JULIA_SCRIPT

    mkdir -p logs

    echo "Submitting ${TOTAL_TASKS} tasks: mode=${MODE}, repeats=${NREPEATS}, L=(${L_LIST[*]}), p=(${P_LIST[*]})"
    sbatch --export=ALL --array="0-$((TOTAL_TASKS - 1))%${MAX_CONCURRENT}" "$0" "${MODE}" "${NREPEATS}" "${MAX_CONCURRENT}"
    exit 0
fi

module load julia

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
export OUT_ADJ="_p${PVAL}_L${LVAL}_rep${REPEAT_INDEX}"

echo "MODE=${MODE} L=${LVAL} p=${PVAL} repeat=${REPEAT_INDEX}"

julia "${JULIA_SCRIPT}"
