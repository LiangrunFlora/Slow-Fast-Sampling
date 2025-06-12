

accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks bbh --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=-1,gen_interval_steps=-1,transfer_ratio=0.0,cache_order=0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
--num_fewshot 3  \
--output_path ./bbh_log \
--log_samples \
--trust_remote_code \

# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks bbh --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
# --gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0 "  \
# --num_fewshot 3  \
# --output_path ./bbh_log \
# --log_samples \
# --trust_remote_code \


# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks bbh --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=25,gen_interval_steps=6,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 3  \
# --output_path ./bbh_log \
# --log_samples \
# --trust_remote_code \



# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks bbh --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=50,gen_interval_steps=6,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 3  \
# --output_path ./bbh_log \
# --log_samples \
# --trust_remote_code \



# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks bbh --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=100,gen_interval_steps=6,cfg_interval_steps=1,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=256,gen_length=256,steps=256,cfg_scale=0.0 "  \
# --num_fewshot 3  \
# --output_path ./bbh_log \
# --log_samples \
# --trust_remote_code \


