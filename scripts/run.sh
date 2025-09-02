#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# container: docker.io/cphsieh/ruler:0.1.0
# bash run.sh RUN_ID BENCHMARK_NAME

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> $1 <benchmark_name>"
    exit 1
fi


# Root Directories
GPUS=$(nvidia-smi --list-gpus | wc -l) # Use all available GPUs
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=${BATCH_SIZE:-8}  # increase to improve GPU utilization
VLLM_MODEL_IMPL=${VLLM_MODEL_IMPL:-auto}  # increase to improve GPU utilization

# NOTE: @goon - bypass the config_models.sh logic and assume the user has specified the following
# directly
# - MODEL_PATH
# - MODEL_TEMPLATE_TYPE
# - MODEL_FRAMEWORK
# - TOKENIZER_PATH
# - TOKENIZER_TYPE
# - ROOT_DIR
#
# RUN_ID is only used to determine where the outputs go


# NOTE: @goon - move the non-MODEL_SELECT code from config_models.sh here
RUN_ID=${1}
TEMPERATURE=${TEMPERATURE:-"0.0"} # greedy by default
TOP_P=${TOP_P:-"1.0"}
TOP_K=${TOP_K:-"32"}

# Turn a comma-separated SEQ_LENGTHS list into a bash array
SEQ_LENGTHS=${SEQ_LENGTHS:-4096,8192,16384,32768,65536,131072}
IFS=',' read -ra SEQ_LENGTHS <<< "$SEQ_LENGTHS"

# # Model and Tokenizer
# NOTE: @goon - still sourcing config_models.sh because it's where we also
# source config_models.sh
# MODEL_CONFIG=$(MODEL_SELECT ${RUN_ID} ${MODEL_DIR} ${ENGINE_DIR})
# IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
# if [ -z "${MODEL_PATH}" ]; then
#     echo "Model: ${RUN_ID} is not supported"
#     exit 1
# fi



export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}


# NOTE: @goon - instead of sourcing config_tasks.sh, replicate the logic here and
# make the tasks we run configurable.

BENCHMARK=synthetic
DEFAULT_TASKS=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2
TASKS=${TASKS:-$DEFAULT_TASKS}

NUM_SAMPLES=${NUM_SAMPLES:-500}
REMOVE_NEWLINE_TAB=${REMOVE_NEWLINE_TAB:-false}
STOP_WORDS=${STOP_WORDS:-""}

if [ -z "${STOP_WORDS}" ]; then
    STOP_WORDS=""
else
    STOP_WORDS="--stop_words \"${STOP_WORDS}\""
fi

if [ "${REMOVE_NEWLINE_TAB}" = false ]; then
    REMOVE_NEWLINE_TAB=""
else
    REMOVE_NEWLINE_TAB="--remove_newline_tab"
fi


echo "BATCH_SIZE=$BATCH_SIZE"
echo "ENGINE_DIR=$ENGINE_DIR"
echo "GPUS=$GPUS"
echo "MODEL_FRAMEWORK=$MODEL_FRAMEWORK"
echo "MODEL_PATH=$MODEL_PATH"
echo "MODEL_TEMPLATE_TYPE=$MODEL_TEMPLATE_TYPE"
echo "NUM_SAMPLES=$NUM_SAMPLES"
echo "ROOT_DIR=$ROOT_DIR"
echo "RUNNING with:"
echo "RUN_ID=$RUN_ID"
echo "TASKS=$TASKS"
echo "TASKS=$TASKS"
echo "TEMPERATURE=$TEMPERATURE"
echo "TOKENIZER_PATH=$TOKENIZER_PATH"
echo "TOKENIZER_TYPE=$TOKENIZER_TYPE"
echo "TOP_K=$TOP_K"
echo "TOP_P=$TOP_P"
echo "VLLM_MODEL_IMPL=$VLLM_MODEL_IMPL"
echo "VLLM_USE_V1=$VLLM_USE_V1"


# Turns TASKS into a list:
IFS=',' read -ra TASKS <<< "$TASKS"


# Start server (you may want to run in other container.)
if [ "$MODEL_FRAMEWORK" == "vllm" ]; then
    python pred/serve_vllm.py \
        --model=${MODEL_PATH} \
        --tensor-parallel-size=${GPUS} \
        --dtype bfloat16 \
        --disable-custom-all-reduce \
        --trust-remote-code \
        --model_impl $VLLM_MODEL_IMPL\
        &

elif [ "$MODEL_FRAMEWORK" == "trtllm" ]; then
    python pred/serve_trt.py \
        --model_path=${MODEL_PATH} \
        &

elif [ "$MODEL_FRAMEWORK" == "sglang" ]; then
    python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --tp ${GPUS} \
        --port 5000 \
        --enable-flashinfer \
        &
    # use sglang/test/killall_sglang.sh to kill sglang server if it hangs

fi


# Start client (prepare data / call model API / obtain final metrics)
total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do

    RESULTS_DIR="${ROOT_DIR}/${RUN_ID}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    echo "Saving MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH} results to ${RESULTS_DIR}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}

    for TASK in "${TASKS[@]}"; do
        python data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}

        start_time=$(date +%s)
        python pred/call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --model_name_or_path ${MODEL_PATH} \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P} \
            --batch_size ${BATCH_SIZE} \
            ${STOP_WORDS}
        end_time=$(date +%s)
        time_diff=$((end_time - start_time))
        total_time=$((total_time + time_diff))
    done

    python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done

echo "Total time spent on call_api: $total_time seconds"
