#!/bin/bash
#SBATCH --job-name=mp_decoder
#SBATCH --partition=caslake
#SBATCH --account=pi-liangjiang
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-11
#SBATCH --output=logs/mp_decoder_%A_%a.out
#SBATCH --error=logs/mp_decoder_%A_%a.err

module load julia

mkdir -p logs

P_LIST=(0.005 0.010 0.015 0.020)
L_LIST=(7 9 13)

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

julia 2d_windowed_simulation_batch.jl