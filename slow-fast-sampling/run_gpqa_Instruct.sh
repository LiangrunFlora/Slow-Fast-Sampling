export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=ERROR


accelerate launch --config_file eval_config.yaml  evaluation_script.py -m lm_eval --model LLADA --tasks gpqa_main_generative_n_shot  --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=64,gen_length=128,steps=128,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--confirm_run_unsafe_code \
--trust_remote_code \
--apply_chat_template \
--fewshot_as_multiturn \





accelerate launch --config_file eval_config.yaml  evaluation_script.py -m lm_eval --model LLADA --tasks gpqa_main_generative_n_shot --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=100,gen_interval_steps=2,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=64,gen_length=128,steps=128,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--confirm_run_unsafe_code \
--trust_remote_code \
--apply_chat_template \
--fewshot_as_multiturn \

accelerate launch --config_file eval_config.yaml  evaluation_script.py -m lm_eval --model LLADA --tasks gpqa_main_generative_n_shot --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=100,gen_interval_steps=4,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=64,gen_length=128,steps=128,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--confirm_run_unsafe_code \
--trust_remote_code \
--apply_chat_template \
--fewshot_as_multiturn \



accelerate launch --config_file eval_config.yaml  evaluation_script.py -m lm_eval --model LLADA --tasks gpqa_main_generative_n_shot --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=100,gen_interval_steps=6,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=64,gen_length=128,steps=128,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--confirm_run_unsafe_code \
--trust_remote_code \
--apply_chat_template \
--fewshot_as_multiturn \




accelerate launch --config_file eval_config.yaml  evaluation_script.py -m lm_eval --model LLADA --tasks gpqa_main_generative_n_shot --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=2,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=64,gen_length=128,steps=128,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--confirm_run_unsafe_code \
--trust_remote_code \
--apply_chat_template \
--fewshot_as_multiturn \

accelerate launch --config_file eval_config.yaml  evaluation_script.py -m lm_eval --model LLADA --tasks gpqa_main_generative_n_shot --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=4,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=64,gen_length=128,steps=128,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--confirm_run_unsafe_code \
--trust_remote_code \
--apply_chat_template \
--fewshot_as_multiturn \



accelerate launch --config_file eval_config.yaml  evaluation_script.py -m lm_eval --model LLADA --tasks gpqa_main_generative_n_shot --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=6,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=64,gen_length=128,steps=128,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--confirm_run_unsafe_code \
--trust_remote_code \
--apply_chat_template \
--fewshot_as_multiturn \




accelerate launch --config_file eval_config.yaml  evaluation_script.py -m lm_eval --model LLADA --tasks gpqa_main_generative_n_shot --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=25,gen_interval_steps=2,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=64,gen_length=128,steps=128,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--confirm_run_unsafe_code \
--trust_remote_code \
--apply_chat_template \
--fewshot_as_multiturn \

accelerate launch --config_file eval_config.yaml  evaluation_script.py -m lm_eval --model LLADA --tasks gpqa_main_generative_n_shot --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=25,gen_interval_steps=4,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=64,gen_length=128,steps=128,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--confirm_run_unsafe_code \
--trust_remote_code \
--apply_chat_template \
--fewshot_as_multiturn \



accelerate launch --config_file eval_config.yaml  evaluation_script.py -m lm_eval --model LLADA --tasks gpqa_main_generative_n_shot --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=25,gen_interval_steps=6,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=64,gen_length=128,steps=128,cfg_scale=0.0,remasking="low_confidence" "  \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--confirm_run_unsafe_code \
--trust_remote_code \
--apply_chat_template \
--fewshot_as_multiturn \
