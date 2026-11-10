#!/bin/bash
set -euo pipefail

# Coarse hub initial scan for cnot_primitive_xz.jl (plans/xz_noise_hub_cnot.md,
# Stage 2): locates the threshold of a gap k before its full scan.
#
# For each gap in GAPS, submits jobs/cnot_primitive_xz_hub_full_scan.sh with
# initial-scan settings: L = 5 7 9, N = L, a wide coarse p grid, and 1000 hub
# failures or 10^4 trials per point. Results go to
# results/cnot_primitive_xz/hub/k_<k>/NL/. The jobs get their own name, log
# files, and time limit through sbatch's SBATCH_JOB_NAME, SBATCH_OUTPUT,
# SBATCH_ERROR, and SBATCH_TIMELIMIT, which override the full-scan script's
# #SBATCH lines.
#
# Task size: one point per task, 4 CPUs (THREADS_PER_TASK), at most 10^4
# trials. Measured 9-11.5 ms per trial on 8 threads of the development machine
# at L = 9 for k = 2-10; with cluster cores up to 2x slower, the slowest task
# takes about 8 min plus Julia startup, so the default 30 min limit keeps a
# margin. Raise it (SBATCH_TIMELIMIT=hh:mm:ss) for much larger L, k, or MAX_TRIALS.
#
# Submit from projects/mp-decoder:
#   bash jobs/cnot_primitive_xz_hub_initial_scan.sh
#   GAPS="2 4" bash jobs/cnot_primitive_xz_hub_initial_scan.sh
#
# Afterwards, copy the results back and locate each threshold with
# hub_initial_scan_summary in notebooks/analysis_utils.py; it prints the pairwise
# crossings and the P_LIST for the full scan.
# Thresholds already located (local initial scans): k = 0 ~0.35%, 1 ~0.53%, 3 ~0.55-0.6%.

GAPS=${GAPS:-"2 5 10"}
INITIAL_P_LIST=${P_LIST:-"0.0015 0.002 0.0025 0.003 0.0035 0.004 0.005 0.006 0.007 0.008 0.010 0.012"}
INITIAL_L_LIST=${L_LIST:-"5 7 9"}

export SBATCH_JOB_NAME=cnot_primitive_xz_hub_initial_scan
export SBATCH_TIMELIMIT="${SBATCH_TIMELIMIT:-00:30:00}"
export SBATCH_OUTPUT="logs/cnot_primitive_xz_hub_initial_scan_%A_%a.out"
export SBATCH_ERROR="logs/cnot_primitive_xz_hub_initial_scan_%A_%a.err"

for k in ${GAPS//,/ }; do
    NREPEATS=1 THREADS_PER_TASK="${THREADS_PER_TASK:-4}" GAP_ROUNDS="${k}" P_LIST="${INITIAL_P_LIST}" L_LIST="${INITIAL_L_LIST}" CNOT_COUNTS=L \
    STOP_MODE=capped ACC_ERRORS="${ACC_ERRORS:-1000}" MAX_TRIALS="${MAX_TRIALS:-10000}" STOP_COUNT=hub \
        bash "$(dirname "$0")/cnot_primitive_xz_hub_full_scan.sh" "$@"
done
