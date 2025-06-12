export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=ERROR


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 2 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=7,cfg_interval_steps=1,transfer_ratio=0.01,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 2 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=7,cfg_interval_steps=1,transfer_ratio=0.1,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 2 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=7,cfg_interval_steps=1,transfer_ratio=0.2,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 2 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=7,cfg_interval_steps=1,transfer_ratio=0.25,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \



# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 2 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=7,cfg_interval_steps=1,transfer_ratio=0.3,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 2 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=7,cfg_interval_steps=1,transfer_ratio=0.4,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 2 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=7,cfg_interval_steps=1,transfer_ratio=0.5,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 2 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=7,cfg_interval_steps=1,transfer_ratio=0.6,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \


accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=7,cfg_interval_steps=1,transfer_ratio=0.0,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
--num_fewshot 4  \
--output_path ./gsm8k_log \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 2 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=7,cfg_interval_steps=1,transfer_ratio=0.7,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 2 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=7,cfg_interval_steps=1,transfer_ratio=0.8,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 2 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=7,cfg_interval_steps=1,transfer_ratio=0.9,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \

# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 2 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=7,cfg_interval_steps=1,transfer_ratio=0.99,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \

# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 2 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,cache_order=0,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \


