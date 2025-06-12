# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=6,gen_interval_steps=2,cache_order=0,is_cache=False" \
# --gen_kwargs "block_length=8,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \
# --apply_chat_template \
# --fewshot_as_multiturn \




accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model Dream --tasks gsm8k --batch_size 1 \
--model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,parallelize=False,backend="causal",mc_num=128,,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "max_new_tokens=256,output_history=True,return_dict_in_generate=True,steps=256,temperature=0.2,top_p=0.95,alg="entropy",alg_temp=0.0" \
--num_fewshot 4  \
--output_path ./gsm8k_log \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \




accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model Dream --tasks gpqa --batch_size 1 \
--model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,parallelize=False,backend="causal",mc_num=128,,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "max_new_tokens=256,output_history=True,return_dict_in_generate=True,steps=256,temperature=0.2,top_p=0.95,alg="entropy",alg_temp=0.0" \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \
--confirm_run_unsafe_code \


accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model Dream --tasks mbpp --batch_size 1 \
--model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,parallelize=False,backend="causal",mc_num=128,,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "max_new_tokens=256,output_history=True,return_dict_in_generate=True,steps=256,temperature=0.2,top_p=0.95,alg="entropy",alg_temp=0.0" \
--num_fewshot 3  \
--output_path ./mbpp_log \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \
--confirm_run_unsafe_code \


accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model Dream --tasks truthfulqa_gen --batch_size 1 \
--model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,parallelize=False,backend="causal",mc_num=128,,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "max_new_tokens=256,output_history=True,return_dict_in_generate=True,steps=256,temperature=0.2,top_p=0.95,alg="entropy",alg_temp=0.0" \
--output_path ./truthfulqa_gen_log \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \
--confirm_run_unsafe_code \














accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model Dream --tasks gsm8k --batch_size 1 \
--model_args "pretrained=Dream-org/Dream-v0-Base-7B,parallelize=False,backend="causal",mc_num=128,,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "max_new_tokens=256,output_history=True,return_dict_in_generate=True,steps=256,temperature=0.2,top_p=0.95,alg="entropy",alg_temp=0.0" \
--num_fewshot 4  \
--output_path ./gsm8k_log \


accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model Dream --tasks gpqa --batch_size 1 \
--model_args "pretrained=Dream-org/Dream-v0-Base-7B,parallelize=False,backend="causal",mc_num=128,,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "max_new_tokens=256,output_history=True,return_dict_in_generate=True,steps=256,temperature=0.2,top_p=0.95,alg="entropy",alg_temp=0.0" \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--confirm_run_unsafe_code \


accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model Dream --tasks mbpp --batch_size 1 \
--model_args "pretrained=Dream-org/Dream-v0-Base-7B,parallelize=False,backend="causal",mc_num=128,,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "max_new_tokens=256,output_history=True,return_dict_in_generate=True,steps=256,temperature=0.2,top_p=0.95,alg="entropy",alg_temp=0.0" \
--num_fewshot 3  \
--output_path ./mbpp_log \
--log_samples \
--confirm_run_unsafe_code \




accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model Dream --tasks truthfulqa_gen --batch_size 1 \
--model_args "pretrained=Dream-org/Dream-v0-Base-7B,parallelize=False,backend="causal",mc_num=128,,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "max_new_tokens=256,output_history=True,return_dict_in_generate=True,steps=256,temperature=0.2,top_p=0.95,alg="entropy",alg_temp=0.0" \
--output_path ./truthfulqa_gen_log \
--log_samples \
--confirm_run_unsafe_code \











accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model Dream --tasks mmlu_pro --batch_size 1 \
--model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,parallelize=False,backend="causal",mc_num=128,,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "max_new_tokens=256,output_history=True,return_dict_in_generate=True,steps=256,temperature=0.2,top_p=0.95,alg="entropy",alg_temp=0.0" \
--num_fewshot 0  \
--output_path ./mmlu_pro_log \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \
--confirm_run_unsafe_code \