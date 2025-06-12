import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import defaultdict
from calflops import calculate_flops


def convert_to_float(value):
    """Convert calflops string (e.g., '500 GFLOPS' or '7.68 TMACs') to float (in FLOPS or MACs)."""
    if isinstance(value, str):
        value = value.strip()
        if "GFLOPS" in value:
            return float(value.replace("GFLOPS", "")) * 1e9
        elif "TFLOPS" in value:
            return float(value.replace("TFLOPS", "")) * 1e12
        elif "MFLOPS" in value:
            return float(value.replace("MFLOPS", "")) * 1e6
        elif "FLOPS" in value:
            return float(value.replace("FLOPS", ""))
        elif "GMACs" in value:
            return float(value.replace("GMACs", "")) * 1e9
        elif "TMACs" in value:
            return float(value.replace("TMACs", "")) * 1e12
        elif "MMACs" in value:
            return float(value.replace("MMACs", "")) * 1e6
        elif "MACs" in value:
            return float(value.replace("MACs", ""))
        else:
            return float(value)
    return float(value)

def parse_args():
    parser = argparse.ArgumentParser(description="Speed and FLOPs test for LLaMA-3.1-8B-Instruct generation")
    parser.add_argument("--prompt_length", type=int, default=893, help="Length of input prompt in tokens")
    parser.add_argument("--generate_length", type=int, default=256, help="Number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--num_trials", type=int, default=3, help="Number of test trials")
    parser.add_argument("--steps", type=int, default=3, help="Number of FLOPs calculation steps")
    parser.add_argument("--prompt_interval_steps", type=int, default=0, help="Prompt interval steps for FLOPs")
    parser.add_argument("--gen_interval_steps", type=int, default=0, help="Generation interval steps for FLOPs")
    parser.add_argument("--transfer_ratio", type=float, default=1.0, help="Transfer ratio for FLOPs")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Path to LLaMA model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model (cuda or cpu)")
    return parser.parse_args()

def create_prompt(tokenizer, length):
    """Create a dummy prompt with approximately the specified token length."""
    sentence = "This is a test sentence used to simulate a long prompt for the LLaMA model. "
    num_repeats = length // len(tokenizer.encode(sentence)) + 1
    prompt = sentence * num_repeats
    tokens = tokenizer.encode(prompt)[:length]
    return tokenizer.decode(tokens)

def main():
    args = parse_args()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # Create prompt
    prompt = create_prompt(tokenizer, args.prompt_length)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    batch_input_ids = input_ids.repeat(args.batch_size, 1)

    # Verify prompt length
    actual_prompt_length = batch_input_ids.shape[-1]
    print(f"Actual prompt length: {actual_prompt_length} tokens")

    # Warm-up run
    print("Running warm-up...")
    with torch.no_grad():
        _ = model.generate(
            batch_input_ids,
            max_new_tokens=args.generate_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Speed test
    times = []
    print(f"Running {args.num_trials} trials with batch_size={args.batch_size}, prompt_length={args.prompt_length}, generate_length={args.generate_length}")
    for i in range(args.num_trials):
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                batch_input_ids,
                max_new_tokens=args.generate_length,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        total_tokens = args.generate_length * args.batch_size
        speed = total_tokens / elapsed_time
        print(f"Trial {i+1}: Time = {elapsed_time:.2f}s, Speed = {speed:.2f} tokens/s")

    # Summarize speed results
    avg_time = np.mean(times)
    avg_speed = (args.generate_length * args.batch_size) / avg_time
    print("\nSpeed Test Summary:")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Average speed: {avg_speed:.2f} tokens/s")

    # FLOPs calculation
    print("\n开始计算Flops")
    flops_cached_total = 0
    macs_cached_total = 0
    for i in range(args.steps):
        flops_cached, macs_cached, _ = calculate_flops(
            model=model,
            input_shape=(args.batch_size, args.prompt_length + args.generate_length),
            print_detailed=False,
            print_results=False,
            transformer_tokenizer=tokenizer,
            output_precision=4
        )
        flops_cached_val = convert_to_float(flops_cached)
        macs_cached_val = convert_to_float(macs_cached)
        flops_cached_total += flops_cached_val
        macs_cached_total += macs_cached_val

    # Calculate averages
    avg_flops_cached = flops_cached_total /args.generate_length
    avg_macs_cached = macs_cached_total /args.generate_length

    # Display results
    result = (
        f"\n| Prompt Interval Steps: {args.prompt_interval_steps} | Gen Interval Steps: {args.gen_interval_steps} | Transfer_Ratio {args.transfer_ratio} =================\n"
        f"With Cache - FLOPs: {avg_flops_cached / 1e12:.4f} TFLOPS   MACs: {avg_macs_cached / 1e12:.4f} TMACs \n"
    )
    print(result)

if __name__ == "__main__":
    main()