#!/bin/bash
#SBATCH --job-name=cnot_primitive_xz_memory_full_scan
#SBATCH --partition=caslake
#SBATCH --account=pi-liangjiang
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --output=logs/cnot_primitive_xz_memory_full_scan_%A_%a.out
#SBATCH --error=logs/cnot_primitive_xz_memory_full_scan_%A_%a.err

set -euo pipefail

# Memory full scan for cnot_primitive_xz.jl (plans/xz_noise_hub_cnot.md):
# the reference for the hub thresholds under the same two-sector noise model.
#
# One block, T = c d noisy rounds with d = L (SCHEDULE=memory), then cleanup and
# readout. T = d is Lake's fixed-aspect observable, the analog of the hub's
# N = c d. One Slurm array task per (repeat, L, p, T); results go to
# ${OUTPUT_DIR}/T<c>d/.
#
# Defaults: T = d and 2d, L = 5-13, a fixed 10^5 trials per point. The p grid is
# set around the Stage-1 threshold estimate (~1.77%): 0.1% steps over
# 1.4-2.3% for the crossing, 0.9% and 1.2% for the suppression factor, 2.7%
# above.
#
# Submit from projects/mp-decoder:
#   bash jobs/cnot_primitive_xz_memory_full_scan.sh [NREPEATS] [MAX_CONCURRENT]
# Every setting below can be overridden from the environment, e.g.
#   L_LIST="15 17 19" bash jobs/cnot_primitive_xz_memory_full_scan.sh

NREPEATS=${1:-${NREPEATS:-1}}
MAX_CONCURRENT=${2:-${MAX_CONCURRENT:-30}}

P_LIST=(${P_LIST:-0.009 0.012 0.014 0.015 0.016 0.017 0.018 0.019 0.020 0.021 0.022 0.023 0.027})
L_LIST=(${L_LIST:-5 7 9 11 13})
T_MULTIPLES=(${T_MULTIPLES:-1 2})   # T = c d
QRAT=${QRAT:-1}
RVAL=${RVAL:-3}
LOGZ=${LOGZ:-true}
CLEANUP_TIME_DEFAULT=${CLEANUP_TIME:-auto}
TRIAL_PARALLEL=${TRIAL_PARALLEL:-true}
MAX_TRIALS=${MAX_TRIALS:-100000}
SEED_BASE=${SEED_BASE:-}
JULIA_MODULE=${JULIA_MODULE:-julia}
THREADS_PER_TASK=${THREADS_PER_TASK:-${SLURM_CPUS_PER_TASK:-16}}
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
PROJECT_DIR=${PROJECT_DIR:-${SUBMIT_DIR}}
JULIA_SCRIPT=${JULIA_SCRIPT:-cnot_primitive_xz.jl}
OUTPUT_DIR=${OUTPUT_DIR:-results/cnot_primitive_xz/memory}

[[ "${PROJECT_DIR}" == /* ]] || PROJECT_DIR="${SUBMIT_DIR}/${PROJECT_DIR}"
[[ "${JULIA_SCRIPT}" == /* ]] || JULIA_SCRIPT="${PROJECT_DIR}/${JULIA_SCRIPT}"
[[ "${OUTPUT_DIR}" == /* ]] || OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"

for c in "${T_MULTIPLES[@]}"; do
    [[ "${c}" =~ ^[1-9][0-9]*$ ]] || { echo "T_MULTIPLES entries must be positive integers; got '${c}'." >&2; exit 1; }
done
[[ "${NREPEATS}" =~ ^[1-9][0-9]*$ ]] || { echo "NREPEATS must be a positive integer." >&2; exit 1; }
[[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_CONCURRENT must be a positive integer." >&2; exit 1; }

NP=${#P_LIST[@]}
NL=${#L_LIST[@]}
NT=${#T_MULTIPLES[@]}
NCOMBOS_PER_REPEAT=$((NP * NL * NT))
TOTAL_TASKS=$((NCOMBOS_PER_REPEAT * NREPEATS))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    P_STR="${P_LIST[*]}"
    L_STR="${L_LIST[*]}"
    T_STR="${T_MULTIPLES[*]}"
    unset P_LIST L_LIST T_MULTIPLES   # arrays cannot be exported; pass the lists as strings
    export P_LIST="${P_STR}" L_LIST="${L_STR}" T_MULTIPLES="${T_STR}"
    export NREPEATS MAX_CONCURRENT QRAT RVAL LOGZ TRIAL_PARALLEL MAX_TRIALS SEED_BASE
    export JULIA_MODULE THREADS_PER_TASK PROJECT_DIR JULIA_SCRIPT OUTPUT_DIR
    export CLEANUP_TIME="${CLEANUP_TIME_DEFAULT}"
    mkdir -p logs "${OUTPUT_DIR}"
    echo "Submitting ${TOTAL_TASKS} memory tasks: L=(${L_STR}) p=(${P_STR}) T=(${T_STR}) x d"
    echo "repeats=${NREPEATS}, threads/task=${THREADS_PER_TASK}, max concurrent=${MAX_CONCURRENT}"
    echo "stop: ${MAX_TRIALS} trials per point; QRAT=${QRAT}"
    echo "output dir=${OUTPUT_DIR}"
    sbatch --cpus-per-task="${THREADS_PER_TASK}" --export=ALL \
        --array="0-$((TOTAL_TASKS - 1))%${MAX_CONCURRENT}" "$0" "${NREPEATS}" "${MAX_CONCURRENT}"
    exit 0
fi

if type module >/dev/null 2>&1; then
    module load "${JULIA_MODULE}"
else
    echo "Environment modules unavailable; using Julia already on PATH."
fi

(( SLURM_ARRAY_TASK_ID < TOTAL_TASKS )) || { echo "Task ${SLURM_ARRAY_TASK_ID} is outside ${TOTAL_TASKS}; exiting."; exit 0; }

T_INDEX=$((SLURM_ARRAY_TASK_ID % NT))
P_INDEX=$(((SLURM_ARRAY_TASK_ID / NT) % NP))
L_INDEX=$(((SLURM_ARRAY_TASK_ID / (NT * NP)) % NL))
REPEAT_INDEX=$((SLURM_ARRAY_TASK_ID / NCOMBOS_PER_REPEAT))

C_VAL=${T_MULTIPLES[$T_INDEX]}
P_VAL=${P_LIST[$P_INDEX]}
L_VAL=${L_LIST[$L_INDEX]}
TASK_DIR="${OUTPUT_DIR}/T${C_VAL}d"
mkdir -p "${TASK_DIR}"

export SCHEDULE=memory LVAL="${L_VAL}" PVAL="${P_VAL}" TVAL="${L_VAL}" MEMORY_ROUNDS=$((C_VAL * L_VAL))
export QRAT RVAL LOGZ TRIAL_PARALLEL MAX_TRIALS STOP_MODE=trials
export CLEANUP_TIME="${CLEANUP_TIME_DEFAULT}" OUTPUT_DIR="${TASK_DIR}"
if (( NREPEATS > 1 )); then export OUT_ADJ="_rep${REPEAT_INDEX}"; else unset OUT_ADJ; fi
if [[ -n "${SEED_BASE}" ]]; then export SEED=$((SEED_BASE + SLURM_ARRAY_TASK_ID)); else unset SEED; fi

echo "task ${SLURM_ARRAY_TASK_ID}: L=${L_VAL} p=${P_VAL} T=${C_VAL}d repeat=${REPEAT_INDEX} -> ${TASK_DIR}"
julia --startup-file=no --threads="${THREADS_PER_TASK}" "${JULIA_SCRIPT}"
