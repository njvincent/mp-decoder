#!/bin/bash
#SBATCH --job-name=cnot_sheetcopy_full_scan_threaded
#SBATCH --partition=caslake
#SBATCH --account=pi-liangjiang
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/cnot_sheetcopy_full_scan_threaded_%A_%a.out
#SBATCH --error=logs/cnot_sheetcopy_full_scan_threaded_%A_%a.err

set -euo pipefail

MODE=${1:-CNOT_Ft}
NREPEATS=${2:-5}
MAX_CONCURRENT=${3:-20}
JULIA_SCRIPT=${JULIA_SCRIPT:-2d_windowed_cnot_sheetcopy.jl}
THREADS_PER_TASK=${THREADS_PER_TASK:-${SLURM_CPUS_PER_TASK:-8}}
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
PROJECT_DIR=${PROJECT_DIR:-${SUBMIT_DIR}}
OUTPUT_DIR=${OUTPUT_DIR:-results/cnot_sheetcopy/full_scan/T∕2_CNOT_T∕2_2T}

if [[ "${PROJECT_DIR}" != /* ]]; then
    PROJECT_DIR="${SUBMIT_DIR}/${PROJECT_DIR}"
fi

if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi

if [[ "${JULIA_SCRIPT}" != /* ]]; then
    JULIA_SCRIPT="${PROJECT_DIR}/${JULIA_SCRIPT}"
fi

P_LIST=(${P_LIST:-0.009 0.010 0.011})
L_LIST=(${L_LIST:-19})
QRAT=${QRAT:-1}
RVAL=${RVAL:-3}
SYNCH=${SYNCH:-true}
LOGZ=${LOGZ:-true}
TRIAL_PARALLEL=${TRIAL_PARALLEL:-true}
CNOT_STYLE=${CNOT_STYLE:-sheetcopy}
ACC_ERRORS=${ACC_ERRORS:-1000}
SAMPS=${SAMPS:-0}
TVAL_DEFAULT=${TVAL:-L}
CLEANUP_TIME_DEFAULT=${CLEANUP_TIME:-auto}
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
    export CNOT_STYLE
    export ACC_ERRORS
    export SAMPS
    export TVAL="${TVAL_DEFAULT}"
    export CLEANUP_TIME="${CLEANUP_TIME_DEFAULT}"
    export JULIA_SCRIPT
    export THREADS_PER_TASK
    export JULIA_MODULE
    export PROJECT_DIR
    export OUTPUT_DIR

    mkdir -p logs
    mkdir -p "${OUTPUT_DIR}"

    echo "Submitting ${TOTAL_TASKS} sheet-copy CNOT full-scan tasks"
    echo "mode=${MODE}, repeats=${NREPEATS}, threads/task=${THREADS_PER_TASK}, max concurrent=${MAX_CONCURRENT}"
    echo "L=(${L_LIST[*]}), p=(${P_LIST[*]}), qrat=${QRAT}, r=${RVAL}, synch=${SYNCH}, logZ=${LOGZ}"
    echo "TVAL=${TVAL_DEFAULT}, CLEANUP_TIME=${CLEANUP_TIME_DEFAULT}, ACC_ERRORS=${ACC_ERRORS}, SAMPS=${SAMPS}"
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

PVAL=${P_LIST[$P_INDEX]}
LVAL=${L_LIST[$L_INDEX]}

if [[ "${TVAL_DEFAULT}" == "L" ]]; then
    TVAL_VAL=${LVAL}
else
    TVAL_VAL=${TVAL_DEFAULT}
fi

T_PRE_VAL=$((TVAL_VAL / 2))
T_POST_VAL=$((TVAL_VAL - T_PRE_VAL))

if [[ "${CLEANUP_TIME_DEFAULT}" == "auto" ]]; then
    CLEANUP_TIME_VAL=$((2 * TVAL_VAL))
else
    CLEANUP_TIME_VAL=${CLEANUP_TIME_DEFAULT}
fi

if (( SAMPS > 0 )); then
    SAMPLE_ADJ="_samps${SAMPS}"
else
    SAMPLE_ADJ="_acc${ACC_ERRORS}"
fi

export PVAL
export LVAL
export MODE
export QRAT
export RVAL
export SYNCH
export LOGZ
export REPEAT_INDEX
export TRIAL_PARALLEL
export CNOT_STYLE
export ACC_ERRORS
export SAMPS
export TVAL="${TVAL_VAL}"
export CLEANUP_TIME="${CLEANUP_TIME_VAL}"
export JULIA_NUM_THREADS="${THREADS_PER_TASK}"
export OUT_ADJ="_cnot_sheetcopy_full_p${PVAL}_L${LVAL}_rep${REPEAT_INDEX}_T${TVAL_VAL}_Tpre${T_PRE_VAL}_Tpost${T_POST_VAL}${SAMPLE_ADJ}_thr${JULIA_NUM_THREADS}"

echo "MODE=${MODE} L=${LVAL} p=${PVAL} repeat=${REPEAT_INDEX}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "JULIA_SCRIPT=${JULIA_SCRIPT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TASK_OUTPUT_DIR=${TASK_OUTPUT_DIR}"
echo "JULIA_NUM_THREADS=${JULIA_NUM_THREADS} TRIAL_PARALLEL=${TRIAL_PARALLEL}"
echo "CNOT_STYLE=${CNOT_STYLE} TVAL=${TVAL} T_PRE=${T_PRE_VAL} T_POST=${T_POST_VAL} CLEANUP_TIME=${CLEANUP_TIME} ACC_ERRORS=${ACC_ERRORS} SAMPS=${SAMPS}"

(
    cd "${TASK_OUTPUT_DIR}"
    julia --threads="${JULIA_NUM_THREADS}" "${JULIA_SCRIPT}"
)
