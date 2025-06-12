accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks humaneval --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=25,gen_interval_steps=8,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False,add_bos_token=True" \
--gen_kwargs "block_length=32,gen_length=512,steps=512,cfg_scale=0.0 "  \
--output_path ./humaneval_log/ \
--log_samples \
--confirm_run_unsafe_code \


# 这个版本是对的？？
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks humaneval --batch_size 2 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=25,gen_interval_steps=8,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False,add_bos_token=True" \
--gen_kwargs "block_length=32,gen_length=512,steps=512,cfg_scale=0.0 "  \
--output_path ./humaneval_log/ \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \
--confirm_run_unsafe_code \

