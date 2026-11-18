#!/bin/bash
#SBATCH --job-name=cnot_primitive_xz_hub_full_scan
#SBATCH --partition=caslake
#SBATCH --account=pi-liangjiang
#SBATCH --time=03:00:00
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
# to ${OUTPUT_DIR}/k_<k>/N<c>L/ ("NL" for c = 1). Noise convention: p is the
# two-qubit gate error rate (P2_RATIO = 1), single-qubit noise 0.1p, q = p
# (QRAT = 1); results/cnot_primitive_xz/hub/ holds the previous convention.
#
# Defaults: k = 0, 1, 2, 5, 10 and L = 5, 7, 9, 13, 19. GAP_ROUNDS and L_LIST take
# lists; each gap, and within it the sizes L <= 13 and L > 13, is submitted as
# its own Slurm array with its own default p grid, chunking and time limit.
# N = L and 2L, a fixed 10^5 trials per point (STOP_MODE=trials),
# split into NREPEATS tasks of MAX_TRIALS = 10^5 / NREPEATS each (independent
# seeds, files _rep<r>; the analysis loader sums them).
# Each default p grid is centered on the gap's threshold estimate p_c, the
# degree-2 finite-size fit to its current-convention initial scan (L = 5, 7, 9;
# docs/implementation.md Section 5.2.11) rounded to 0.02%: k = 0 1.48%,
# k = 1 1.84%, k = 2 2.08%, k = 5 2.26%, k = 10 2.36%. Steps of 0.02% over
# p_c +- 0.2%, plus p_c +- 0.3% and +- 0.5% (25 values): the fit region
# p_c +- 0.1% holds 5 points on each side, and at least 4 remain if the
# full-scan threshold lies within 0.14% of p_c. Any other k needs P_LIST. To
# add points to an existing scan, pass only the new ones in P_LIST:
# resubmitting a default grid reruns every point.
#
# Task size, chosen to keep jobs short for backfill and a gap under 1000 tasks:
# 16 CPUs, and NREPEATS = 2 when every L <= 13, 4 otherwise, so a gap's default
# grid is 400 + 200 = 600 tasks. All five default gaps together are 3000 tasks;
# under a per-user job limit, submit them separately (GAP_ROUNDS=k). The
# slowest point (N = 2L) measured about 88 ms per trial at L = 13 and 450 ms at
# L = 19 on 8 threads of the development machine; with twice the threads and
# trials per task and cluster cores up to 2x slower, the slowest task takes
# about 75 min (L = 13, 5e4 trials) and 3.1 h (L = 19, 2.5e4 trials).
# The time limit is 3 h when every L <= 13 and 8 h otherwise, scaled with the
# trials and threads per task if those are changed (at most 36 h);
# SBATCH_TIMELIMIT=hh:mm:ss overrides it.
#
# Submit from projects/mp-decoder:
#   bash jobs/cnot_primitive_xz_hub_full_scan.sh [NREPEATS] [MAX_CONCURRENT]
# Every setting below can be overridden from the environment, e.g.
#   GAP_ROUNDS=1 bash jobs/cnot_primitive_xz_hub_full_scan.sh
#   GAP_ROUNDS="1 3" bash jobs/cnot_primitive_xz_hub_full_scan.sh
# With several arrays, MAX_CONCURRENT applies to each array, and P_LIST (if
# given) to every gap.
#   GAP_ROUNDS=2 P_LIST="0.019 0.020 0.021 0.022" L_LIST="5 7 9" CNOT_COUNTS=L bash jobs/cnot_primitive_xz_hub_full_scan.sh

NREPEATS=${1:-${NREPEATS:-auto}}   # tasks per point; auto: 2 if every L <= 13, else 4
MAX_CONCURRENT=${2:-${MAX_CONCURRENT:-30}}

default_p_grid() {   # a gap's default p grid; empty if its threshold is not measured
    case "$1" in
        0) echo "0.0098 0.0118 0.0128 0.013 0.0132 0.0134 0.0136 0.0138 0.014 0.0142 0.0144 0.0146 0.0148 0.015 0.0152 0.0154 0.0156 0.0158 0.016 0.0162 0.0164 0.0166 0.0168 0.0178 0.0198" ;;
        1) echo "0.0134 0.0154 0.0164 0.0166 0.0168 0.017 0.0172 0.0174 0.0176 0.0178 0.018 0.0182 0.0184 0.0186 0.0188 0.019 0.0192 0.0194 0.0196 0.0198 0.02 0.0202 0.0204 0.0214 0.0234" ;;
        2) echo "0.0158 0.0178 0.0188 0.019 0.0192 0.0194 0.0196 0.0198 0.02 0.0202 0.0204 0.0206 0.0208 0.021 0.0212 0.0214 0.0216 0.0218 0.022 0.0222 0.0224 0.0226 0.0228 0.0238 0.0258" ;;
        5) echo "0.0176 0.0196 0.0206 0.0208 0.021 0.0212 0.0214 0.0216 0.0218 0.022 0.0222 0.0224 0.0226 0.0228 0.023 0.0232 0.0234 0.0236 0.0238 0.024 0.0242 0.0244 0.0246 0.0256 0.0276" ;;
        10) echo "0.0186 0.0206 0.0216 0.0218 0.022 0.0222 0.0224 0.0226 0.0228 0.023 0.0232 0.0234 0.0236 0.0238 0.024 0.0242 0.0244 0.0246 0.0248 0.025 0.0252 0.0254 0.0256 0.0266 0.0286" ;;
    esac
}

GAP_ROUNDS=${GAP_ROUNDS:-"0 1 2 5 10"}
GAP_LIST=(${GAP_ROUNDS//,/ })
L_SPEC=${L_LIST:-"5 7 9 13 19"}
L_ALL=(${L_SPEC//,/ })
SMALL_L=(); LARGE_L=()               # L <= 13: 4 chunks, 3 h; L > 13: 8 chunks, 8 h
for L in "${L_ALL[@]}"; do
    [[ "${L}" =~ ^[1-9][0-9]*$ ]] || { echo "L_LIST entries must be positive integers; got '${L}'." >&2; exit 1; }
    if (( L <= 13 )); then SMALL_L+=("${L}"); else LARGE_L+=("${L}"); fi
done
L_CLASSES=()
(( ${#SMALL_L[@]} > 0 )) && L_CLASSES+=("${SMALL_L[*]}")
(( ${#LARGE_L[@]} > 0 )) && L_CLASSES+=("${LARGE_L[*]}")
for k in "${GAP_LIST[@]}"; do        # check every gap before submitting any
    [[ "${k}" =~ ^[0-9]+$ ]] ||
        { echo "GAP_ROUNDS entries must be nonnegative integers; got '${k}'." >&2; exit 1; }
    [[ -n "${P_LIST:-$(default_p_grid "${k}")}" ]] ||
        { echo "No default p grid for GAP_ROUNDS=${k} (threshold not measured yet); set P_LIST." >&2; exit 1; }
done
if (( ${#GAP_LIST[@]} * ${#L_CLASSES[@]} > 1 )) && [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    for k in "${GAP_LIST[@]}"; do      # one submission (and array) per gap and size class
        for sizes in "${L_CLASSES[@]}"; do
            GAP_ROUNDS="${k}" L_LIST="${sizes}" bash "$0" "$@"
        done
    done
    exit 0
fi
P_LIST=(${P_LIST:-$(default_p_grid "${GAP_ROUNDS}")})
L_LIST=("${L_ALL[@]}")
MAX_L=$(printf '%s\n' "${L_LIST[@]}" | sort -n | tail -1)
[[ "${NREPEATS}" == "auto" ]] && { if (( MAX_L <= 13 )); then NREPEATS=2; else NREPEATS=4; fi; }
CNOT_COUNTS_NORMALIZED=${CNOT_COUNTS:-"L 2L"}
CNOT_COUNTS_NORMALIZED=${CNOT_COUNTS_NORMALIZED//,/ }
N_LIST=()
read -r -a N_LIST <<< "${CNOT_COUNTS_NORMALIZED}"
QRAT=${QRAT:-1}
P2_RATIO=${P2_RATIO:-1}
RVAL=${RVAL:-3}
LOGZ=${LOGZ:-true}
CLEANUP_TIME_DEFAULT=${CLEANUP_TIME:-auto}
TRACK_FRAMES=${TRACK_FRAMES:-true}
TRIAL_PARALLEL=${TRIAL_PARALLEL:-true}
STOP_MODE=${STOP_MODE:-trials}    # standard: a fixed MAX_TRIALS per point
STOP_COUNT=${STOP_COUNT:-hub}     # failures ACC_ERRORS counts: hub (target A) or any (failed trials)
ACC_ERRORS=${ACC_ERRORS:-2000}
TOTAL_TRIALS=${TOTAL_TRIALS:-100000}   # per point, summed over its NREPEATS tasks
[[ "${NREPEATS}" =~ ^[1-9][0-9]*$ ]] || { echo "NREPEATS must be a positive integer or auto." >&2; exit 1; }
MAX_TRIALS=${MAX_TRIALS:-$(( (TOTAL_TRIALS + NREPEATS - 1) / NREPEATS ))}   # per task
SEED_BASE=${SEED_BASE:-}
JULIA_MODULE=${JULIA_MODULE:-julia}
THREADS_PER_TASK=${THREADS_PER_TASK:-${SLURM_CPUS_PER_TASK:-16}}
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
PROJECT_DIR=${PROJECT_DIR:-${SUBMIT_DIR}}
JULIA_SCRIPT=${JULIA_SCRIPT:-cnot_primitive_xz.jl}
OUTPUT_DIR=${OUTPUT_DIR:-results/cnot_primitive_xz/p_2q/hub}

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
    # limit: 3 h per 2.5e4 trials (L <= 13) or 8 h per 1.25e4 trials (L > 13) at 8
    # threads, scaled by this task's trials and threads, at most 36 h
    if (( MAX_L <= 13 )); then BASE_H=3; BASE_TRIALS=25000; else BASE_H=8; BASE_TRIALS=12500; fi
    HOURS=$(( (BASE_H * MAX_TRIALS * 8 + BASE_TRIALS * THREADS_PER_TASK - 1) / (BASE_TRIALS * THREADS_PER_TASK) ))
    (( HOURS > 36 )) && HOURS=36
    if [[ -n "${SBATCH_TIMELIMIT:-}" ]]; then TIME_OPT=""; TIME_STR="${SBATCH_TIMELIMIT}"
    else TIME_STR=$(printf '%02d:00:00' "${HOURS}"); TIME_OPT="--time=${TIME_STR}"; fi
    unset P_LIST L_LIST        # arrays cannot be exported; pass the lists as strings
    export P_LIST="${P_STR}" L_LIST="${L_STR}" CNOT_COUNTS="${N_STR}"
    export NREPEATS MAX_CONCURRENT GAP_ROUNDS QRAT P2_RATIO RVAL LOGZ TRACK_FRAMES TRIAL_PARALLEL
    export STOP_MODE STOP_COUNT ACC_ERRORS MAX_TRIALS TOTAL_TRIALS SEED_BASE JULIA_MODULE THREADS_PER_TASK PROJECT_DIR JULIA_SCRIPT OUTPUT_DIR
    export CLEANUP_TIME="${CLEANUP_TIME_DEFAULT}"
    mkdir -p logs "${OUTPUT_DIR}"
    echo "Submitting ${TOTAL_TASKS} hub tasks: L=(${L_STR}) p=(${P_STR}) N=(${N_STR}) k=${GAP_ROUNDS}"
    echo "repeats=${NREPEATS}, threads/task=${THREADS_PER_TASK}, max concurrent=${MAX_CONCURRENT}, time limit/task=${TIME_STR}"
    if [[ "${STOP_MODE}" == "trials" ]]; then STOP_STR="${MAX_TRIALS} trials per task, $(( MAX_TRIALS * NREPEATS )) per point"
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
