

accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks truthfulqa_gen --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=100,gen_interval_steps=10,cache_order=0,transfer_ratio=0.25,is_cache=True" \
--gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
--output_path ./truthfulqa_gen_log \
--log_samples \
--trust_remote_code

