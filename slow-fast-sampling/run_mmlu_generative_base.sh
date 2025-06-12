


accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks mmlu_generative --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=3,gen_length=3,steps=3,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./mmlu_generative_log \
--log_samples \


accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks mmlu_generative --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=100,gen_interval_steps=4,cfg_interval_steps=1,transfer_ratio=0.26,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=3,gen_length=3,steps=3,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./mmlu_generative_log \
--log_samples \



accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks mmlu_generative --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=4,cfg_interval_steps=1,transfer_ratio=0.26,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=3,gen_length=3,steps=3,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./mmlu_generative_log \
--log_samples \



accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks mmlu_generative --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=25,gen_interval_steps=4,cfg_interval_steps=1,transfer_ratio=0.26,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=3,gen_length=3,steps=3,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./mmlu_generative_log \
--log_samples \





accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks mmlu_generative --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=100,gen_interval_steps=6,cfg_interval_steps=1,transfer_ratio=0.26,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=3,gen_length=3,steps=3,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./mmlu_generative_log \
--log_samples \



accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks mmlu_generative --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=6,cfg_interval_steps=1,transfer_ratio=0.26,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=3,gen_length=3,steps=3,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./mmlu_generative_log \
--log_samples \



accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks mmlu_generative --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=25,gen_interval_steps=6,cfg_interval_steps=1,transfer_ratio=0.26,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=3,gen_length=3,steps=3,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./mmlu_generative_log \
--log_samples \