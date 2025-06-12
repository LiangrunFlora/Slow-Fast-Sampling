#!/bin/bash

# 如果任何命令失败，则立即退出
set -e
# 将未设置的变量视为错误
set -u
# 管道命令的返回值是最后一个失败命令的返回值
set -o pipefail

export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=ERROR

echo "开始评估运行 1/16: cycle_len_confidence_threshold=0.1, high_confidence_threshold=0.9"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.1,high_confidence_threshold=0.9"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.1_hct0.9 \
--log_samples
echo "运行 1 完成。"
echo "----------------------------------------------------"

echo "开始评估运行 2/16: cycle_len_confidence_threshold=0.15, high_confidence_threshold=0.9"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.15,high_confidence_threshold=0.9"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.15_hct0.9 \
--log_samples
echo "运行 2 完成。"
echo "----------------------------------------------------"

echo "开始评估运行 3/16: cycle_len_confidence_threshold=0.2, high_confidence_threshold=0.9"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.2,high_confidence_threshold=0.9"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.2_hct0.9 \
--log_samples
echo "运行 3 完成。"
echo "----------------------------------------------------"

echo "开始评估运行 4/16: cycle_len_confidence_threshold=0.25, high_confidence_threshold=0.9"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.25,high_confidence_threshold=0.9"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.25_hct0.9 \
--log_samples
echo "运行 4 完成。"
echo "----------------------------------------------------"

echo "开始评估运行 5/16: cycle_len_confidence_threshold=0.3, high_confidence_threshold=0.9"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.9"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.3_hct0.9 \
--log_samples
echo "运行 5 完成。"
echo "----------------------------------------------------"

echo "开始评估运行 6/16: cycle_len_confidence_threshold=0.35, high_confidence_threshold=0.9"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.35,high_confidence_threshold=0.9"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.35_hct0.9 \
--log_samples
echo "运行 6 完成。"
echo "----------------------------------------------------"

echo "开始评估运行 7/16: cycle_len_confidence_threshold=0.4, high_confidence_threshold=0.9"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.4,high_confidence_threshold=0.9"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.4_hct0.9 \
--log_samples
echo "运行 7 完成。"
echo "----------------------------------------------------"

echo "开始评估运行 8/16: cycle_len_confidence_threshold=0.45, high_confidence_threshold=0.9"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.45,high_confidence_threshold=0.9"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.45_hct0.9 \
--log_samples
echo "运行 8 完成。"
echo "----------------------------------------------------"

echo "开始评估运行 9/16: cycle_len_confidence_threshold=0.5, high_confidence_threshold=0.9"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.5,high_confidence_threshold=0.9"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.5_hct0.9 \
--log_samples
echo "运行 9 完成。"
echo "----------------------------------------------------"

echo "开始评估运行 10/16: cycle_len_confidence_threshold=0.3, high_confidence_threshold=0.7"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.7"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.3_hct0.7 \
--log_samples
echo "运行 10 完成。"
echo "----------------------------------------------------"

echo "开始评估运行 11/16: cycle_len_confidence_threshold=0.3, high_confidence_threshold=0.75"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.75"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.3_hct0.75 \
--log_samples
echo "运行 11 完成。"
echo "----------------------------------------------------"

echo "开始评估运行 12/16: cycle_len_confidence_threshold=0.3, high_confidence_threshold=0.8"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.8"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.3_hct0.8 \
--log_samples
echo "运行 12 完成。"
echo "----------------------------------------------------"

echo "开始评估运行 13/16: cycle_len_confidence_threshold=0.3, high_confidence_threshold=0.85"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.85"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.3_hct0.85 \
--log_samples
echo "运行 13 完成。"
echo "----------------------------------------------------"

# echo "开始评估运行 14/16: cycle_len_confidence_threshold=0.3, high_confidence_threshold=0.9 (与运行5参数相同)"
# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
# --gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.9"  \
# --num_fewshot 4  \
# --output_path ./hyper_log/gsm8k_clct0.3_hct0.9_run14 \
# --log_samples
# echo "运行 14 完成。"
# echo "----------------------------------------------------"

echo "开始评估运行 15/16: cycle_len_confidence_threshold=0.3, high_confidence_threshold=0.95"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.95"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.3_hct0.95 \
--log_samples
echo "运行 15 完成。"
echo "----------------------------------------------------"

echo "开始评估运行 16/16: cycle_len_confidence_threshold=0.3, high_confidence_threshold=1.0"
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=1.0"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_clct0.3_hct1.0 \
--log_samples
echo "运行 16 完成。"
echo "----------------------------------------------------"

echo "所有评估已完成。"