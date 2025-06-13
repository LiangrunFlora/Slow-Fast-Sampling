#!/bin/bash

model="Dream-org/Dream-v0-Base-7B"

export HF_ALLOW_CODE_EVAL=1

ACCEL_CONFIG="eval_config.yaml"
MAIN_PORT="29510" 

echo "Starting evaluation for minerva_math"

# --- Task Specific Parameters for minerva_math ---
TASK="minerva_math"
NUM_FEWSHOT=4     # From tasks="... minerva_math ...", nshots="... 4 ..."
MAX_NEW_TOKENS=256 # From tasks="... minerva_math ...", lengths="... 512 ..."
DIFFUSION_STEPS=256 # Note: based on original script (equal to max_new_tokens)
TEMPERATURE=0.2    # From tasks="... minerva_math ...", temperatures="... 0 ..."
TOP_P=0.95        # Constant in the original loop's model_args
ADD_BOS_TOKEN="true" # Constant in the original loop's model_args
# Note: original loop did NOT include escape_until=true

OUTPUT_PATH="./${TASK}_log"

accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN},is_feature_cache=True,is_cfg_cache=False \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --confirm_run_unsafe_code

echo "Completed evaluation for ${TASK}"