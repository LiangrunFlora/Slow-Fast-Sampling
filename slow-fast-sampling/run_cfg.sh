
# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.2,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.15"  \
# --num_fewshot 4  \
# --output_path ./cfg_log/gsm8k \
# --log_samples \


python evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=True" \
--gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.15"  \
--num_fewshot 4  \
--output_path ./cfg_log/gsm8k \
--log_samples \
--limit 1



# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=-1,gen_interval_steps=-1,transfer_ratio=0.0,cache_order=0,is_cache=False" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.50 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=-1,gen_interval_steps=-1,transfer_ratio=0.0,cache_order=0,is_cache=False" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=1.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=-1,gen_interval_steps=-1,transfer_ratio=0.0,cache_order=0,is_cache=False" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=2.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \

