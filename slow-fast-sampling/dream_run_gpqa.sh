#!/bin/bash

# 设置模型
model="Dream-org/Dream-v0-Base-7B"

# 允许代码评估 (如果任务需要，保留)
export HF_ALLOW_CODE_EVAL=1

# Accelerate 配置和主进程端口
ACCEL_CONFIG="eval_config.yaml"
MAIN_PORT="29510" # 使用与 humaneval 相同的默认端口

echo "Starting evaluation for gpqa_main_generative_n_shot"

# --- Task Specific Parameters for gpqa_main_generative_n_shot ---
TASK="gpqa_main_generative_n_shot"
NUM_FEWSHOT=5     # From tasks="... gpqa_main_generative_n_shot ...", nshots="... 4 ..."
MAX_NEW_TOKENS=256 # From tasks="... gpqa_main_generative_n_shot ...", lengths="... 512 ..."
DIFFUSION_STEPS=256 # Note: based on original script (equal to max_new_tokens)
TEMPERATURE=0.0    # From tasks="... gpqa_main_generative_n_shot ...", temperatures="... 0 ..."
TOP_P=0.95        # Constant in the original loop's model_args
ADD_BOS_TOKEN="true" # Constant in the original loop's model_args
# Note: original loop did NOT include escape_until=true

# 输出路径
OUTPUT_PATH="./${TASK}_log"

# 执行评估命令
accelerate launch --config_file ${ACCEL_CONFIG} --main_process_port ${MAIN_PORT} evaluation_script.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${DIFFUSION_STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=${ADD_BOS_TOKEN} \
    --tasks ${TASK} \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --output_path ${OUTPUT_PATH} \
    --log_samples \
    --confirm_run_unsafe_code

echo "Completed evaluation for ${TASK}"