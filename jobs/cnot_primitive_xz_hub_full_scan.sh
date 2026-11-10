#!/bin/bash
#SBATCH --job-name=cnot_primitive_xz_hub_full_scan
#SBATCH --partition=caslake
#SBATCH --account=pi-liangjiang
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --output=logs/cnot_primitive_xz_hub_full_scan_%A_%a.out
#SBATCH --error=logs/cnot_primitive_xz_hub_full_scan_%A_%a.err

set -euo pipefail

# Hub full scan for cnot_primitive_xz.jl (plans/xz_noise_hub_cnot.md, Stage 2).
#
# Design: the hub A (target) does N noisy CNOT rounds with fresh partners,
# GAP_ROUNDS = k ordinary rounds apart, with the same k at every size, and
# N = c d CNOTs with d = L. The gate density 1/(k+1) and the circuit's aspect
# ratio are then fixed. One Slurm array task per (repeat, L, p, N); results go
# to ${OUTPUT_DIR}/k_<k>/N<c>L/ ("NL" for c = 1).
#
# Defaults: k = 1, N = L and 2L, a fixed 10^5 trials per point (STOP_MODE=trials).
# Each default p grid spans about 0.5-1.5 times the threshold, in steps of
# about 5-7% of it near p_c. Measured thresholds: k = 0 ~0.35% and k = 3
# ~0.55-0.6% (local initial scans, L = 5-9), k = 1 ~0.53%. Any other k needs P_LIST.
#
# Time limit per task: the slowest point (N = 2L, 10^5 trials) measured about
# 88 ms per trial at L = 13 and 450 ms at L = 19 on 8 threads of the development
# machine. On 16 threads with cluster cores up to 2x slower that is about 2.5 h
# and 12.5 h. The limit is 6 h when every L <= 13 and 24 h otherwise;
# SBATCH_TIMELIMIT=hh:mm:ss overrides it.
#
# Submit from projects/mp-decoder:
#   bash jobs/cnot_primitive_xz_hub_full_scan.sh [NREPEATS] [MAX_CONCURRENT]
# Every setting below can be overridden from the environment, e.g.
#   GAP_ROUNDS=3 bash jobs/cnot_primitive_xz_hub_full_scan.sh
#   GAP_ROUNDS=2 P_LIST="0.004 0.005 0.006 0.007" L_LIST="5 7 9" CNOT_COUNTS=L bash jobs/cnot_primitive_xz_hub_full_scan.sh

NREPEATS=${1:-${NREPEATS:-1}}
MAX_CONCURRENT=${2:-${MAX_CONCURRENT:-30}}

GAP_ROUNDS=${GAP_ROUNDS:-1}
[[ "${GAP_ROUNDS}" =~ ^[0-9]+$ ]] ||
    { echo "GAP_ROUNDS must be a nonnegative integer; got '${GAP_ROUNDS}'." >&2; exit 1; }
case "${GAP_ROUNDS}" in
    0) P_DEFAULT="0.002 0.0025 0.00275 0.003 0.00325 0.0035 0.00375 0.004 0.00425 0.0045 0.005" ;;
    1) P_DEFAULT="0.0025 0.003 0.0035 0.004 0.0045 0.005 0.0055 0.006 0.0065 0.007 0.008" ;;
    3) P_DEFAULT="0.003 0.0035 0.004 0.0045 0.005 0.0055 0.006 0.0065 0.007 0.0075 0.0085" ;;
    *) P_DEFAULT="" ;;
esac
[[ -n "${P_LIST:-${P_DEFAULT}}" ]] ||
    { echo "No default p grid for GAP_ROUNDS=${GAP_ROUNDS} (threshold not measured yet); set P_LIST." >&2; exit 1; }
P_LIST=(${P_LIST:-${P_DEFAULT}})
L_LIST=(${L_LIST:-5 7 9 11 13})
CNOT_COUNTS_NORMALIZED=${CNOT_COUNTS:-"L 2L"}
CNOT_COUNTS_NORMALIZED=${CNOT_COUNTS_NORMALIZED//,/ }
N_LIST=()
read -r -a N_LIST <<< "${CNOT_COUNTS_NORMALIZED}"
QRAT=${QRAT:-1}
P2_RATIO=${P2_RATIO:-10}
RVAL=${RVAL:-3}
LOGZ=${LOGZ:-true}
CLEANUP_TIME_DEFAULT=${CLEANUP_TIME:-auto}
TRACK_FRAMES=${TRACK_FRAMES:-true}
TRIAL_PARALLEL=${TRIAL_PARALLEL:-true}
STOP_MODE=${STOP_MODE:-trials}    # standard: a fixed MAX_TRIALS per point
STOP_COUNT=${STOP_COUNT:-hub}     # failures ACC_ERRORS counts: hub (target A) or any (failed trials)
ACC_ERRORS=${ACC_ERRORS:-2000}
MAX_TRIALS=${MAX_TRIALS:-100000}
SEED_BASE=${SEED_BASE:-}
JULIA_MODULE=${JULIA_MODULE:-julia}
THREADS_PER_TASK=${THREADS_PER_TASK:-${SLURM_CPUS_PER_TASK:-16}}
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
PROJECT_DIR=${PROJECT_DIR:-${SUBMIT_DIR}}
JULIA_SCRIPT=${JULIA_SCRIPT:-cnot_primitive_xz.jl}
OUTPUT_DIR=${OUTPUT_DIR:-results/cnot_primitive_xz/hub}

[[ "${PROJECT_DIR}" == /* ]] || PROJECT_DIR="${SUBMIT_DIR}/${PROJECT_DIR}"
[[ "${JULIA_SCRIPT}" == /* ]] || JULIA_SCRIPT="${PROJECT_DIR}/${JULIA_SCRIPT}"
[[ "${OUTPUT_DIR}" == /* ]] || OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"

for n in "${N_LIST[@]}"; do
    [[ "${n}" =~ ^([1-9][0-9]*)?L$ ]] ||
        { echo "CNOT_COUNTS entries must be multiples of L (L, 2L, ...); got '${n}'." >&2; exit 1; }
done
[[ "${NREPEATS}" =~ ^[1-9][0-9]*$ ]] || { echo "NREPEATS must be a positive integer." >&2; exit 1; }
[[ "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_CONCURRENT must be a positive integer." >&2; exit 1; }
case "${STOP_MODE}" in
    trials|failures|capped) ;;
    *) echo "STOP_MODE must be trials, failures, or capped." >&2; exit 1 ;;
esac
case "${STOP_COUNT}" in
    any|hub) ;;
    *) echo "STOP_COUNT must be any or hub." >&2; exit 1 ;;
esac

NP=${#P_LIST[@]}
NL=${#L_LIST[@]}
NN=${#N_LIST[@]}
NCOMBOS_PER_REPEAT=$((NP * NL * NN))
TOTAL_TASKS=$((NCOMBOS_PER_REPEAT * NREPEATS))
(( TOTAL_TASKS >= 1 )) || { echo "No tasks to submit." >&2; exit 1; }

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    P_STR="${P_LIST[*]}"
    L_STR="${L_LIST[*]}"
    N_STR="${N_LIST[*]}"
    MAX_L=$(printf '%s\n' "${L_LIST[@]}" | sort -n | tail -1)
    if [[ -n "${SBATCH_TIMELIMIT:-}" ]]; then TIME_OPT=""; TIME_STR="${SBATCH_TIMELIMIT}"
    elif (( MAX_L <= 13 )); then TIME_OPT="--time=06:00:00"; TIME_STR="06:00:00"
    else TIME_OPT="--time=24:00:00"; TIME_STR="24:00:00"; fi
    unset P_LIST L_LIST        # arrays cannot be exported; pass the lists as strings
    export P_LIST="${P_STR}" L_LIST="${L_STR}" CNOT_COUNTS="${N_STR}"
    export NREPEATS MAX_CONCURRENT GAP_ROUNDS QRAT P2_RATIO RVAL LOGZ TRACK_FRAMES TRIAL_PARALLEL
    export STOP_MODE STOP_COUNT ACC_ERRORS MAX_TRIALS SEED_BASE JULIA_MODULE THREADS_PER_TASK PROJECT_DIR JULIA_SCRIPT OUTPUT_DIR
    export CLEANUP_TIME="${CLEANUP_TIME_DEFAULT}"
    mkdir -p logs "${OUTPUT_DIR}"
    echo "Submitting ${TOTAL_TASKS} hub tasks: L=(${L_STR}) p=(${P_STR}) N=(${N_STR}) k=${GAP_ROUNDS}"
    echo "repeats=${NREPEATS}, threads/task=${THREADS_PER_TASK}, max concurrent=${MAX_CONCURRENT}, time limit/task=${TIME_STR}"
    if [[ "${STOP_MODE}" == "trials" ]]; then STOP_STR="${MAX_TRIALS} trials per point"
    else STOP_STR="${STOP_MODE} on ${STOP_COUNT} failures (ACC_ERRORS=${ACC_ERRORS}, MAX_TRIALS=${MAX_TRIALS})"; fi
    echo "stop: ${STOP_STR}; P2_RATIO=${P2_RATIO}, QRAT=${QRAT}"
    echo "output dir=${OUTPUT_DIR}"
    sbatch --cpus-per-task="${THREADS_PER_TASK}" ${TIME_OPT} --export=ALL \
        --array="0-$((TOTAL_TASKS - 1))%${MAX_CONCURRENT}" "$0" "${NREPEATS}" "${MAX_CONCURRENT}"
    exit 0
fi

if type module >/dev/null 2>&1; then
    module load "${JULIA_MODULE}"
else
    echo "Environment modules unavailable; using Julia already on PATH."
fi

(( SLURM_ARRAY_TASK_ID < TOTAL_TASKS )) || { echo "Task ${SLURM_ARRAY_TASK_ID} is outside ${TOTAL_TASKS}; exiting."; exit 0; }

N_INDEX=$((SLURM_ARRAY_TASK_ID % NN))
P_INDEX=$(((SLURM_ARRAY_TASK_ID / NN) % NP))
L_INDEX=$(((SLURM_ARRAY_TASK_ID / (NN * NP)) % NL))
REPEAT_INDEX=$((SLURM_ARRAY_TASK_ID / NCOMBOS_PER_REPEAT))

N_ENTRY=${N_LIST[$N_INDEX]}
P_VAL=${P_LIST[$P_INDEX]}
L_VAL=${L_LIST[$L_INDEX]}
N_COEF=${N_ENTRY%L}
N_VAL=$(( ${N_COEF:-1} * L_VAL ))     # N = c d with d = L
TASK_DIR="${OUTPUT_DIR}/k_${GAP_ROUNDS}/N${N_ENTRY}"
mkdir -p "${TASK_DIR}"

export SCHEDULE=hub LVAL="${L_VAL}" PVAL="${P_VAL}" CNOT_COUNT="${N_VAL}" GAP_ROUNDS TVAL="${L_VAL}"
export QRAT P2_RATIO RVAL LOGZ TRACK_FRAMES TRIAL_PARALLEL STOP_MODE STOP_COUNT ACC_ERRORS MAX_TRIALS
export CLEANUP_TIME="${CLEANUP_TIME_DEFAULT}" OUTPUT_DIR="${TASK_DIR}"
if (( NREPEATS > 1 )); then export OUT_ADJ="_rep${REPEAT_INDEX}"; else unset OUT_ADJ; fi
if [[ -n "${SEED_BASE}" ]]; then export SEED=$((SEED_BASE + SLURM_ARRAY_TASK_ID)); else unset SEED; fi

echo "task ${SLURM_ARRAY_TASK_ID}: L=${L_VAL} p=${P_VAL} N=${N_VAL} k=${GAP_ROUNDS} repeat=${REPEAT_INDEX} -> ${TASK_DIR}"
julia --startup-file=no --threads="${THREADS_PER_TASK}" "${JULIA_SCRIPT}"
