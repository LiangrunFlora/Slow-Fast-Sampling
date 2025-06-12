
export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=ERROR

accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks mmlu_pro --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=-1,gen_interval_steps=-1,cache_order=-1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./gsm8k_log \
--log_samples \

# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks mmlu_pro --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=100,gen_interval_steps=7,cache_order=0,transfer_ratio=0.25,is_cache=True" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 0  \
# --output_path ./gsm8k_log \
# --log_samples \


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks mmlu_pro --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=7,cache_order=0,transfer_ratio=0.25,is_cache=True" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 0  \
# --output_path ./gsm8k_log \
# --log_samples \

