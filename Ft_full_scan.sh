#!/bin/bash
#SBATCH --job-name=mp_decoder
#SBATCH --partition=caslake
#SBATCH --account=pi-liangjiang
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-44%15
#SBATCH --output=logs/mp_decoder_%A_%a.out
#SBATCH --error=logs/mp_decoder_%A_%a.err

module load julia

mkdir -p logs

P_LIST=(0.011 0.012 0.013 0.014 0.015 0.016 0.017 0.018 0.019)
L_LIST=(5 7 9 13 19)

NP=${#P_LIST[@]}
NL=${#L_LIST[@]}

P_INDEX=$((SLURM_ARRAY_TASK_ID % NP))
L_INDEX=$((SLURM_ARRAY_TASK_ID / NP))

PVAL=${P_LIST[$P_INDEX]}
LVAL=${L_LIST[$L_INDEX]}

export PVAL
export LVAL
export MODE=Ft
export QRAT=1
export RVAL=3
export SYNCH=true
export LOGZ=true
export OUT_ADJ="_p${PVAL}_L${LVAL}"

julia 2d_windowed_simulation.jl
