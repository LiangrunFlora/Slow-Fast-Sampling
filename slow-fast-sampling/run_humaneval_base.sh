export HF_ALLOW_CODE_EVAL="1"



# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks humaneval --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=2,gen_interval_steps=2,transfer_ratio=0.25,cache_order=0,is_cache=False" \
# --gen_kwargs "block_length=1024,gen_length=1024,steps=1024,cfg_scale=0.0 "  \
# --num_fewshot 0  \
# --output_path ./humaneval_log/ \
# --log_samples \
# --confirm_run_unsafe_code \

# # original版本
# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks humaneval --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,transfer_ratio=0.0,cache_order=-1,is_feature_cache=False,is_cfg_cache=False" \
# --gen_kwargs "block_length=32,gen_length=512,steps=512,cfg_scale=0.0 "  \
# --num_fewshot 0  \
# --output_path ./humaneval_log/ \
# --log_samples \
# --confirm_run_unsafe_code \

# multi_phase版本
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks humaneval --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,transfer_ratio=0.0,cache_order=-1,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=32,gen_length=512,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./humaneval_log/ \
--log_samples \
--confirm_run_unsafe_code \

# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks humaneval --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=49,gen_interval_steps=7,transfer_ratio=0.125,cache_order=0,is_cache=True" \
# --gen_kwargs "block_length=1024,gen_length=1024,steps=1024,cfg_scale=0.0 "  \
# --num_fewshot 0  \
# --output_path ./humaneval_log/ \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \
# --confirm_run_unsafe_code


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks humaneval --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=49,gen_interval_steps=7,transfer_ratio=0.5,cache_order=0,is_cache=True" \
# --gen_kwargs "block_length=1024,gen_length=1024,steps=1024,cfg_scale=0.0 "  \
# --num_fewshot 0  \
# --output_path ./humaneval_log/ \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \
# --confirm_run_unsafe_code






# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks humaneval --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=5,transfer_ratio=0.25,cache_order=0,is_cache=True" \
# --gen_kwargs "block_length=1024,gen_length=1024,steps=1024,cfg_scale=0.0 "  \
# --num_fewshot 0  \
# --output_path ./humaneval_log/ \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \
# --confirm_run_unsafe_code

# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks humaneval --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=5,transfer_ratio=0.125,cache_order=0,is_cache=True" \
# --gen_kwargs "block_length=1024,gen_length=1024,steps=1024,cfg_scale=0.0 "  \
# --num_fewshot 0  \
# --output_path ./humaneval_log/ \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \
# --confirm_run_unsafe_code


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks humaneval --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=5,transfer_ratio=0.5,cache_order=0,is_cache=True" \
# --gen_kwargs "block_length=1024,gen_length=1024,steps=1024,cfg_scale=0.0 "  \
# --num_fewshot 0  \
# --output_path ./humaneval_log/ \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \
# --confirm_run_unsafe_code





# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks humaneval --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=52,gen_interval_steps=4,transfer_ratio=0.25,cache_order=0,is_cache=True" \
# --gen_kwargs "block_length=1024,gen_length=1024,steps=1024,cfg_scale=0.0 "  \
# --num_fewshot 0  \
# --output_path ./humaneval_log/ \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \
# --confirm_run_unsafe_code\


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks humaneval --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=52,gen_interval_steps=4,transfer_ratio=0.125,cache_order=0,is_cache=True" \
# --gen_kwargs "block_length=1024,gen_length=1024,steps=1024,cfg_scale=0.0 "  \
# --num_fewshot 0  \
# --output_path ./humaneval_log/ \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \
# --confirm_run_unsafe_code


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks humaneval --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=52,gen_interval_steps=4,transfer_ratio=0.5,cache_order=0,is_cache=True" \
# --gen_kwargs "block_length=1024,gen_length=1024,steps=1024,cfg_scale=0.0 "  \
# --num_fewshot 0  \
# --output_path ./humaneval_log/ \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \
# --confirm_run_unsafe_code
