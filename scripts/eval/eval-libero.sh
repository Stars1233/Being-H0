#!/bin/bash

# ============================================================
# LIBERO Benchmark Evaluation Script
# ============================================================

RUN_NUM=0
export CUDA_VISIBLE_DEVICES=${RUN_NUM}
export PYTHONPATH=.

# --- Model Configuration ---
MODEL_PATH="<path-to-checkpoint>"           # e.g., /path/to/Being-H05-2B_libero
MODEL_NAME="<checkpoint-name>"              # e.g., Being-H05-2B_libero

# --- Task Suite ---
# Options: "spatial", "object", "goal", "long", "10"
EVAL_SUITES=("spatial" "object" "goal" "long")

# --- Server Configuration ---
SERVER_PORT=1888${RUN_NUM}
SERVER_LOG_FILE="results/eval/logs/${MODEL_NAME}/server_${SERVER_PORT}.log"
mkdir -p "results/eval/logs/${MODEL_NAME}"

SERVER_PID=""
EVAL_PID=""

echo "=============== LIBERO Evaluation ==============="
echo "Model: ${MODEL_PATH}"

# --- Data Configuration ---
DATA_CONFIG_NAME="libero_nonorm"
EMBODIMENT_TAG="libero"
DATASET_NAME="libero_posttrain"
STATS_SELECTION_MODE="task"

# For cross-embodiment models, uncomment and set:
# DATASET_NAME="uni_posttrain"
# METADATA_VARIANT="libero_spatial"  # or specific task name

SERVER_SEED=42
SERVER_PROMPT_TEMP=long
SERVER_MAX_VIEW_NUM=-1

# --- MPG Configuration ---
USE_MPG=True
MPG_LAMBDA=0.1
MPG_NUM_PROJECTIONS=32
MPG_REFINEMENT_ITERS=1
MPG_GATE_TEMPERATURE=2.0

# --- Evaluation Configuration ---
EVAL_SEED=41
EVAL_CHUNK_SIZE=8
EVAL_ACTION_TYPE="world_delta"
EVAL_DATA_CONFIG_NAME="libero"
NUM_TRIALS=50
NUM_SAVE_VIDEOS=3
EVAL_LOG_INTERVAL=10

# --- Helper Functions ---
kill_tree() {
    local _pid=$1
    local _sig=${2:-9}
    if [ -z "$_pid" ]; then return; fi
    local _children=$(pgrep -P "$_pid")
    for _child in $_children; do
        kill_tree "$_child" "$_sig"
    done
    if kill -0 "$_pid" 2>/dev/null; then
        kill -$_sig "$_pid" 2>/dev/null
    fi
}

cleanup() {
    echo ""
    echo "=============== Cleanup ==============="
    if [ -n "$EVAL_PID" ]; then
        kill_tree "$EVAL_PID"
    fi
    if [ -n "$SERVER_PID" ]; then
        kill -9 ${SERVER_PID} 2>/dev/null
    fi
    PIDS_ON_PORT=$(lsof -t -i:${SERVER_PORT} 2>/dev/null)
    if [ -n "$PIDS_ON_PORT" ]; then
        echo "${PIDS_ON_PORT}" | xargs kill -9 2>/dev/null
    fi
}

trap cleanup EXIT INT TERM

echo ""
echo "=============== Step 1: Starting Server ==============="

MPG_ARGS=""
if [ ! -z "${USE_MPG}" ]; then MPG_ARGS="${MPG_ARGS} --use-mpg ${USE_MPG}"; fi
if [ ! -z "${MPG_LAMBDA}" ]; then MPG_ARGS="${MPG_ARGS} --mpg-lambda ${MPG_LAMBDA}"; fi
if [ ! -z "${MPG_NUM_PROJECTIONS}" ]; then MPG_ARGS="${MPG_ARGS} --mpg-num-projections ${MPG_NUM_PROJECTIONS}"; fi
if [ ! -z "${MPG_REFINEMENT_ITERS}" ]; then MPG_ARGS="${MPG_ARGS} --mpg-refinement-iters ${MPG_REFINEMENT_ITERS}"; fi
if [ ! -z "${MPG_GATE_TEMPERATURE}" ]; then MPG_ARGS="${MPG_ARGS} --mpg-gate-temperature ${MPG_GATE_TEMPERATURE}"; fi

RTC_ARGS="--no-enable-rtc"

METADATA_VARIANT_ARGS=""
if [ ! -z "${METADATA_VARIANT}" ]; then
    METADATA_VARIANT_ARGS="${METADATA_VARIANT_ARGS} --metadata-variant ${METADATA_VARIANT}"
fi
METADATA_VARIANT_ARGS="${METADATA_VARIANT_ARGS} --stats-selection-mode ${STATS_SELECTION_MODE}"

nohup python -u -m BeingH.inference.run_server_vla \
    --model-path "${MODEL_PATH}" \
    --port ${SERVER_PORT} \
    --data-config-name "${DATA_CONFIG_NAME}" \
    --dataset-name "${DATASET_NAME}" \
    --embodiment-tag "${EMBODIMENT_TAG}" \
    --seed "${SERVER_SEED}" \
    --prompt-template "${SERVER_PROMPT_TEMP}" \
    --max-view-num $SERVER_MAX_VIEW_NUM \
    --no-use-fixed-view \
    ${MPG_ARGS} ${RTC_ARGS} ${METADATA_VARIANT_ARGS} > "${SERVER_LOG_FILE}" 2>&1 &

SERVER_PID=$!

echo "Waiting for server to be ready..."
MAX_RETRIES=300
COUNTER=0
SERVER_READY=false

while [ $COUNTER -lt $MAX_RETRIES ]; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Error: Server process exited!"
        tail -n 10 "${SERVER_LOG_FILE}"
        exit 1
    fi
    if grep -q "Server is ready" "${SERVER_LOG_FILE}"; then
        echo "Server started successfully!"
        SERVER_READY=true
        break
    fi
    sleep 3
    ((COUNTER++))
done

if [ "$SERVER_READY" = false ]; then
    echo "Error: Server startup timeout."
    exit 1
fi

echo ""
echo "=============== Step 2: Running Evaluation ==============="

for SUITE_NAME in "${EVAL_SUITES[@]}"; do
    echo ">>> Evaluating: libero_${SUITE_NAME}"
    VIDEO_DIR="results/rollouts/${MODEL_NAME}/${SUITE_NAME}_"
    EVAL_LOG_FILE="results/eval/logs/${MODEL_NAME}/libero_${SUITE_NAME}.log"
    mkdir -p "$(dirname ${EVAL_LOG_FILE})"

    python -m BeingH.benchmark.libero.run_libero_eval_fast \
        --task_suite_name "libero_${SUITE_NAME}" \
        --port $SERVER_PORT \
        --seed $EVAL_SEED \
        --video_dir ${VIDEO_DIR} \
        --num_open_loop_steps $EVAL_CHUNK_SIZE \
        --num_trials_per_task $NUM_TRIALS \
        --num_save_videos_per_task $NUM_SAVE_VIDEOS \
        --log_interval $EVAL_LOG_INTERVAL \
        --action_type $EVAL_ACTION_TYPE \
        --data_config_name $EVAL_DATA_CONFIG_NAME 2>&1 | tee "${EVAL_LOG_FILE}"

    echo ">>> Completed: libero_${SUITE_NAME}"
done

echo ""
echo "=============== All Evaluations Complete ==============="
echo "Results saved to: results/eval/logs/${MODEL_NAME}"
