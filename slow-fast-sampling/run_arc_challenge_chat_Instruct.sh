
export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=ERROR
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks arc_challenge_chat --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cache_order=0,is_cache=False" \
--gen_kwargs "block_length=512,gen_length=512,steps=512,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./arc_challenge_chat_log \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \


accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks arc_challenge_chat --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=5,gen_interval_steps=1,cache_order=0,is_cache=True" \
--gen_kwargs "block_length=512,gen_length=512,steps=512,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./arc_challenge_chat_log \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \



accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks arc_challenge_chat --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=10,gen_interval_steps=1,cache_order=0,is_cache=True" \
--gen_kwargs "block_length=512,gen_length=512,steps=512,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./arc_challenge_chat_log \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \


accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks arc_challenge_chat --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=1,cache_order=0,is_cache=True" \
--gen_kwargs "block_length=512,gen_length=512,steps=512,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./arc_challenge_chat_log \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \


accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks arc_challenge_chat --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=100,gen_interval_steps=1,cache_order=0,is_cache=True" \
--gen_kwargs "block_length=512,gen_length=512,steps=512,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./arc_challenge_chat_log \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \
