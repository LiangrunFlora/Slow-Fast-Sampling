
export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=ERROR

# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=-1,gen_interval_steps=-1,transfer_ratio=0.0,cache_order=0,is_cache=False" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.25 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \

# # 对的一版
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,,cfg_interval_steps=1, transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0"  \
--num_fewshot 4  \
--output_path ./cfg_log/gsm8k \
--log_samples \

# # checkboard版本
# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
# --gen_kwargs "block_length=256,gen_length=256,steps=128,cfg_scale=0.0"  \
# --num_fewshot 4  \
# --output_path ./cfg_log/gsm8k \
# --log_samples \

# multi_phase版本
# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,,cfg_interval_steps=1, transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
# --gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0"  \
# --num_fewshot 4  \
# --output_path ./cfg_log/gsm8k \
# --log_samples \


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=-1,gen_interval_steps=-1,transfer_ratio=0.25,cache_order=0,is_cache=False" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \

# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=100,gen_interval_steps=7,transfer_ratio=0.25,cache_order=0,is_cache=True" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \

# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=100,gen_interval_steps=6,transfer_ratio=0.25,cache_order=0,is_cache=True" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \

# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=100,gen_interval_steps=6,transfer_ratio=0.25,cache_order=0,is_cache=True" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
