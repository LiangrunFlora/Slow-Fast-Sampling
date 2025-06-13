#!/bin/bash

model="Dream-org/Dream-v0-Base-7B"

export HF_ALLOW_CODE_EVAL=1

ACCEL_CONFIG="eval_config.yaml"
MAIN_PORT="29510" 

echo "Starting evaluation for gsm8k_cot"

# --- Task Specific Parameters for gsm8k_cot ---
TASK="gsm8k_cot"
NUM_FEWSHOT=8     # From tasks="gsm8k_cot ...", nshots="8 ..."
MAX_NEW_TOKENS=256 # From tasks="gsm8k_cot ...", lengths="256 ..."
DIFFUSION_STEPS=256 # Note: based on original script (equal to max_new_tokens)
TEMPERATURE=0.2    # From tasks="gsm8k_cot ...", temperatures="0 ..."
TOP_P=0.95        # Constant in the original loop's model_args
ADD_BOS_TOKEN="true" # Constant in the original loop's model_args
# Note: original loop did NOT include escape_until=true

OUTPUT_PATH="./${TASK}_log"

accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg="entropy",alg_temp=0.0,is_feature_cache=False,is_cfg_cache=False \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --confirm_run_unsafe_code

echo "Completed evaluation for ${TASK}"