export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks longbench_hotpotqa --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=32,gen_length=32,steps=32,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./longbench_log \
--log_samples \
--apply_chat_template \
--fewshot_as_multiturn \
--trust_remote_code 

