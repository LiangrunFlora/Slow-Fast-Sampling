# python test_speed_flops.py --is_cache --prompt_interval_steps 100 --gen_interval_steps 7 --transfer_ratio 0.25

# python test_speed_flops.py --prompt_interval_steps 1 --gen_interval_steps 1 --transfer_ratio 0.0 --batch_size 1 --gpu_ids 0 1 2 3 --steps 128

python test_speed_flops.py --prompt_interval_steps 1 --gen_interval_steps 1 --transfer_ratio 0.0 --batch_size 1 --gpu_ids 0 1 2 3 4 5 6 7 --gen_length 461 --steps 118 --block_length 461 --avg_prompt_length 146

# python test_speed_flops.py --steps 64 --batch_size 1   

# python test_speed_flops.py --steps 128 --batch_size 1   

# python test_speed_flops.py --steps 256 --batch_size 1   

# python test_speed_flops.py --is_cache --prompt_interval_steps 100 --gen_interval_steps 7 --transfer_ratio 0.15


# python test_speed_flops.py --is_cache --prompt_interval_steps 100 --gen_interval_steps 7 --transfer_ratio 0.35


# python test_speed_flops.py --is_cache --prompt_interval_steps 100 --gen_interval_steps 7 --transfer_ratio 0.45


# python test_speed_flops.py --is_cache --prompt_interval_steps 100 --gen_interval_steps 7 --transfer_ratio 0.55

# python test_speed_flops.py --is_cache --prompt_interval_steps 100 --gen_interval_steps 7 --transfer_ratio 0.65



# python test_speed_flops.py --steps 32 --batch_size 1   --block_length 16

# python test_speed_flops.py --steps 16 --batch_size 1   --block_length 32

# python test_speed_flops.py --steps 8 --batch_size 1  --block_length 64

# python test_speed_flops.py --steps 4 --batch_size 1  --block_length 128


# python test_speed_flops.py --is_cache --prompt_interval_steps 100 --gen_interval_steps 100 --transfer_ratio 0.15