import torch
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy
import torch.nn.functional as F
import numpy as np
import torch
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy
from dllm_cache import FeatureCache
import torch.nn.functional as F
import numpy as np
import accelerate
import time

import matplotlib.pyplot as plt
import seaborn as sns
import os

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits.exp()
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    """
    # mask_index: (batch_size, block_length)，布尔张量，表示掩码位置
    # mask_num: (batch_size, 1)，每行掩码数量
    mask_num = mask_index.sum(dim=1, keepdim=True)
    # base: (batch_size, 1)，每步基础转移token数
    base = mask_num // steps
    # remainder: (batch_size, 1)，剩余token数
    remainder = mask_num % steps
    # num_transfer_tokens: (batch_size, steps)，初始为base扩展到steps列
    num_transfer_tokens = base.expand(-1, steps).clone()
    # 处理remainder
    if remainder.sum() > 0:
        # indices: (steps,)，步数索引
        indices = torch.arange(steps, device=mask_index.device)
        # mask: (batch_size, steps)，布尔张量，标记哪些位置需要+1
        mask = indices.unsqueeze(0) < remainder
        # num_transfer_tokens: (batch_size, steps)，更新后的每步转移token数
        num_transfer_tokens[mask] += 1
    # 返回值: (batch_size, steps)，int64类型，每步转移token数
    return num_transfer_tokens.to(torch.int64)

def generate(
    input_ids,
    attention_mask,
    model,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    plot_confidence_maps=False, # Added flag to control plotting
    plot_every_n_steps=1
):
    with torch.no_grad():
        if plot_confidence_maps:
            save_plot_dir = 'confidence_map'
            if save_plot_dir:
                os.makedirs(save_plot_dir, exist_ok=True)
                print(f"Plots will be saved to: {os.path.abspath(save_plot_dir)}")
            
        batch_size, prompt_length = input_ids.shape
        # Initialize x with mask_id and copy prompt tokens
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )   
        x[:, :prompt_length] = input_ids


        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        feature_cache = FeatureCache()
        feature_cache.reset_cache(prompt_length)
        for num_block in range(num_blocks):
            start_idx = prompt_length + num_block * block_length
            end_idx = prompt_length + (num_block + 1) * block_length

            block_x = x[:, start_idx:end_idx]
            block_mask_index = block_x == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id
                if cfg_scale > 0.0:
                    if hasattr(feature_cache, 'cfg_interval_steps'):
                        feature_cache.update_step(layer_id=33)
                        if feature_cache.refresh_cfg(layer_id=33):
                            cfg_x = x.clone()
                            cfg_x[prompt_index] = mask_id 
                            logits = model(x,attention_mask=attention_mask).logits[:, prompt_length:]
                            feature_cache.cache_type="cfg"
                            cfg_logits = model(cfg_x,attention_mask=attention_mask).logits[:, prompt_length:]
                            cfg_residual = (logits-cfg_logits)
                            feature_cache.set_cache(layer_id=33, feature_name="cfg_residual", features=cfg_residual, cache_type="gen")
                            feature_cache.cache_type="no_cfg"
                        else:
                            feature_cache.cache_type="cfg"
                            cfg_residual = feature_cache.get_cache(layer_id=33, feature_name="cfg_residual", cache_type="gen")
                            feature_cache.cache_type="no_cfg"
                            logits = model(x,attention_mask=attention_mask).logits[:, prompt_length:]
                    else:
                        cfg_x = x.clone()
                        cfg_x[prompt_index] = mask_id 
                        logits = model(x,attention_mask=attention_mask).logits[:, prompt_length:]
                        cfg_logits = model(cfg_x,attention_mask=attention_mask).logits[:, prompt_length:]
                        cfg_residual = (logits-cfg_logits)
                    # logits = cfg_logits + (cfg_scale + 1) * (logits - cfg_logits)
                    logits = (logits-cfg_residual) + (cfg_scale + 1) * cfg_residual
                else:
                    logits = model(x, attention_mask=attention_mask).logits[:, prompt_length:]
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:,(num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index[:, prompt_length:], x0, x[:, prompt_length:])
                confidence = torch.where(mask_index[:, prompt_length:], x0_p, -np.inf)
                
                # --- PLOTTING CONFIDENCE MAP ---
                if plot_confidence_maps and (i % plot_every_n_steps == 0):
                    print(f"    Plotting confidence map for Block {num_block}, Step {i}...")
                    for batch_idx in range(confidence.shape[0]):
                        conf_np = confidence[batch_idx].to(torch.float32).cpu().numpy() # Shape: (gen_length,)
                        
                        # For heatmap, NaNs are often better handled for color scale than -inf
                        conf_for_heatmap = conf_np.copy()
                        is_inf = np.isinf(conf_for_heatmap)
                        conf_for_heatmap[is_inf] = np.nan # Seaborn handles NaNs by not coloring them or using nan_color

                        plt.figure(figsize=(max(12, gen_length // 6), 4)) # Adjust figsize as needed
                        
                        # Plotting the 1D confidence array as a heatmap row
                        # annot_data shows original values including -np.inf
                        # heatmap_data uses NaN for -np.inf for better colormap scaling
                        sns.heatmap(conf_for_heatmap.reshape(1, -1),
                                    annot=conf_np.reshape(1, -1),
                                    fmt=".2f", # Format for annotations
                                    cmap="viridis",
                                    cbar=True,
                                    xticklabels=gen_length//10 if gen_length > 20 else True, # Show some ticks
                                    yticklabels=False, # No y-ticks for a single row
                                    # nan_color='lightgray' # Color for NaN cells (originally -inf)
                                   )
                        
                        title = (f"Confidence Map - Batch {batch_idx}, Block {num_block}, Step {i}\n"
                                 f"(Prompt Len: {prompt_length}, Gen Len: {gen_length}, Block Len: {block_length})")
                        plt.title(title)
                        plt.xlabel("Token Position in Generation")
                        plt.ylabel(f"Batch Item {batch_idx}") # Label for clarity if ever >1 row
                        

                        # current_block_start_gen = num_block * block_length
                        # current_block_end_gen = (num_block + 1) * block_length
                        # plt.axvspan(current_block_start_gen - 0.5, current_block_end_gen - 0.5,
                        #             color='red', alpha=0.1, label=f"Current Block ({num_block})")
                        # if num_block > 0: # Highlight previous blocks
                        #     plt.axvspan(-0.5, current_block_start_gen - 0.5,
                        #                 color='blue', alpha=0.1, label="Previous Blocks")

                        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
                        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
                        if save_plot_dir:
                            filename = f"confidence_b{batch_idx}_block{num_block}_step{i}_iter{steps_per_block}.png"
                            filepath = os.path.join(save_plot_dir, filename)
                            plt.savefig(filepath, dpi=150) # Save the figure
                            plt.close()
                # --- END PLOTTING ---

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i]).indices
                    transfer_index[j, select_index] = True
                x[:, prompt_length:][transfer_index] = x0[transfer_index]

            # if i == 255:
            #     exit(0)
                    
        return x[:, prompt_length:]
