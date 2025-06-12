export ACCELERATE_LOG_LEVEL=error
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=ERROR


accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.9,cycle_length_stability_window=2,cycle_length_stability_std_dev_threshold=1.0"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_window2_std1.0 \
--log_samples

accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.9,cycle_length_stability_window=3,cycle_length_stability_std_dev_threshold=1.0"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_window3_std1.0 \
--log_samples

accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.9,cycle_length_stability_window=4,cycle_length_stability_std_dev_threshold=1.0"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_window4_std1.0 \
--log_samples

accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.9,cycle_length_stability_window=5,cycle_length_stability_std_dev_threshold=1.0"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_window5_std1.0 \
--log_samples

accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.9,cycle_length_stability_window=6,cycle_length_stability_std_dev_threshold=1.0"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_window6_std1.0 \
--log_samples


accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.9,cycle_length_stability_window=2,cycle_length_stability_std_dev_threshold=0.5"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_window2_std0.5 \
--log_samples

accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.9,cycle_length_stability_window=2,cycle_length_stability_std_dev_threshold=1.0"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_window2_std1.0 \
--log_samples

accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.9,cycle_length_stability_window=2,cycle_length_stability_std_dev_threshold=1.5"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_window2_std1.5 \
--log_samples

accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.9,cycle_length_stability_window=2,cycle_length_stability_std_dev_threshold=2.0"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_window2_std2.0 \
--log_samples

accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.9,cycle_length_stability_window=2,cycle_length_stability_std_dev_threshold=2.5"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_window2_std2.5 \
--log_samples

accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.9,cycle_length_stability_window=2,cycle_length_stability_std_dev_threshold=3.0"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_window2_std3.0 \
--log_samples


accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
--model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,backend="causal",mc_num=128,prompt_interval_steps=1,gen_interval_steps=1,cfg_interval_steps=1,transfer_ratio=0.0,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0,k_exploration_steps=6,cycle_len_confidence_threshold=0.3,high_confidence_threshold=0.9,cycle_length_stability_window=2,cycle_length_stability_std_dev_threshold=4.0"  \
--num_fewshot 4  \
--output_path ./hyper_log/gsm8k_window2_std4.0 \
--log_samples

