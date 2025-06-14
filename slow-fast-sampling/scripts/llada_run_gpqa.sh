accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gpqa_main_generative_n_shot --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0 "  \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--trust_remote_code \

accelerate launch --config_file accelerate_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gpqa_main_generative_n_shot --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,prompt_interval_steps=15,gen_interval_steps=5,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0 "  \
--num_fewshot 5  \
--output_path ./gpqa_log \
--log_samples \
--trust_remote_code \