#!/bin/bash
set -euo pipefail

# Coarse hub initial scan for cnot_primitive_xz.jl (plans/xz_noise_hub_cnot.md,
# Stage 2): locates the threshold of a gap k before its full scan.
#
# For each gap in GAPS, submits jobs/cnot_primitive_xz_hub_full_scan.sh with
# initial-scan settings: L = 5 7 9, N = L, a wide coarse p grid, and 1000 hub
# failures or 10^4 trials per point. Results go to
# results/cnot_primitive_xz/hub/k_<k>/NL/. The jobs get their own name and log
# files through sbatch's SBATCH_JOB_NAME, SBATCH_OUTPUT, and SBATCH_ERROR,
# which override the full-scan script's #SBATCH lines.
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
export SBATCH_OUTPUT="logs/cnot_primitive_xz_hub_initial_scan_%A_%a.out"
export SBATCH_ERROR="logs/cnot_primitive_xz_hub_initial_scan_%A_%a.err"

for k in ${GAPS//,/ }; do
    GAP_ROUNDS="${k}" P_LIST="${INITIAL_P_LIST}" L_LIST="${INITIAL_L_LIST}" CNOT_COUNTS=L \
    STOP_MODE=capped ACC_ERRORS="${ACC_ERRORS:-1000}" MAX_TRIALS="${MAX_TRIALS:-10000}" STOP_COUNT=hub \
        bash "$(dirname "$0")/cnot_primitive_xz_hub_full_scan.sh" "$@"
done
