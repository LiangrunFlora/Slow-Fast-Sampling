accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks bbh --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
--num_fewshot 3  \
--output_path ./bbh_log \
--log_samples \
--trust_remote_code \

accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks bbh --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,prompt_interval_steps=5,gen_interval_steps=1,transfer_ratio=0.0,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
--num_fewshot 3  \
--output_path ./bbh_log \
--log_samples \
--trust_remote_code \