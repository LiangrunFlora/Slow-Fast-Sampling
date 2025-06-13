#!/bin/bash

model="Dream-org/Dream-v0-Base-7B"

export HF_ALLOW_CODE_EVAL=1

ACCEL_CONFIG="eval_config.yaml"
MAIN_PORT="29510"

echo "Starting evaluation for humaneval"

# --- Task Specific Parameters for humaneval ---
TASK="humaneval"
NUM_FEWSHOT=0
MAX_NEW_TOKENS=256
DIFFUSION_STEPS=256 # Note: based on original script
TEMPERATURE=0.2
TOP_P=0.95
ADD_BOS_TOKEN="true"
ESCAPE_UNTIL="true" # Note: specific to the humaneval run in original script

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

### NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}