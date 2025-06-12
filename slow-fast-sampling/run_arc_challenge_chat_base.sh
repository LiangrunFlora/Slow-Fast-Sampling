
export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=ERROR
accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks arc_easy --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,transfer_ratio=0.25,cache_order=0,is_cache=False" \
--gen_kwargs "block_length=128,gen_length=128,steps=128,cfg_scale=0.0,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./arc_challenge_chat_log  \
--log_samples \
--limit 8

# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks arc_challenge_chat --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1000,gen_interval_steps=512,transfer_ratio=0.25,cache_order=0,is_cache=True" \
# --gen_kwargs "block_length=1024,gen_length=1024,steps=1024,cfg_scale=0.0 "  \
# --num_fewshot 0  \
# --output_path ./arc_challenge_chat_log  \
# --log_samples \
# --limit 8
