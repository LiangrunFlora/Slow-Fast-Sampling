import collections
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
from sampling_utils.plot_confidence_map import plot_confidence_heatmap

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
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1
    return num_transfer_tokens.to(torch.int64)

def get_num_tokens_for_phase1_step(current_sub_cycle_mask, num_to_fill_this_step_factor=0.1):
    """
    Determines how many tokens to attempt to fill in a single step of Phase 1 exploration.
    Can be a fixed number, or a fraction of remaining masks in the current sub-cycle focus.
    """
    batch_size = current_sub_cycle_mask.shape[0]
    return torch.full((batch_size,), 1, dtype=torch.long, device=current_sub_cycle_mask.device) # try to fill 1 tokens

def get_num_tokens_for_phase3_step(current_sub_cycle_mask, num_to_fill_this_step_factor=0.1):
    """
    Determines how many tokens to attempt to fill in a single step of Phase 1 exploration.
    Can be a fixed number, or a fraction of remaining masks in the current sub-cycle focus.
    """
    batch_size = current_sub_cycle_mask.shape[0]
    return torch.full((batch_size,), 1, dtype=torch.long, device=current_sub_cycle_mask.device) # try to fill 1 tokens

def generate_slow_fast_sampling(
    input_ids,
    attention_mask,
    model,
    # Removed `steps` as it's now implicitly defined by sub-cycle structure
    gen_length=128,
    block_length=128, # Max length of a sub-cycle focus.
    temperature=0.0,
    cfg_scale=0.0,
    mask_id=126336,
    plot_confidence_maps=False,
    plot_save_dir="confidence_plots_multi_phase",
    # Sub-cycle phase parameters
    k_exploration_steps=6,         # Fixed steps for phase 1 (cycle determination)
    cycle_len_confidence_threshold=0.3,
    cycle_length_stability_window=2,
    cycle_length_stability_std_dev_threshold=1.0, # Stricter threshold
    high_confidence_threshold=0.9,
    num_important_low_confidence_tokens=3, # Fewer for more focused fill
    max_sub_cycles_per_block=256 # Safety break for blocks
):
    with torch.no_grad():
        batch_size, prompt_length = input_ids.shape
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id, dtype=torch.long, device=model.device,
        )
        x[:, :prompt_length] = input_ids
        prompt_index_full_x = (x != mask_id)

        assert gen_length % block_length == 0, "gen_length must be divisible by block_length for this simplified block approach"
        num_blocks = gen_length // block_length

        feature_cache = FeatureCache()
        feature_cache.reset_cache(prompt_length,gen_length=gen_length)
        
        total_model_calls = 0 # For tracking computation
        totol_model_gen_length = 0

        for block_idx in range(num_blocks):
            # print(f"\n--- Processing Block {block_idx + 1}/{num_blocks} ---")
            block_abs_start_in_x = prompt_length + block_idx * block_length
            block_abs_end_in_x = prompt_length + (block_idx + 1) * block_length

            current_sub_cycles_in_block = 0
            actual_sub_cycle_length_per_item = torch.full((batch_size,), block_length, dtype=torch.long, device=x.device)
            last_sub_cycle_length_per_item = torch.full((batch_size,), 0, dtype=torch.long, device=x.device)
            # Loop for decoding sub-cycles within the current block
            while True:
                mask_in_current_block_abs_coords = (x[:, block_abs_start_in_x:block_abs_end_in_x] == mask_id)
                if not mask_in_current_block_abs_coords.any():
                    break
                if current_sub_cycles_in_block >= max_sub_cycles_per_block:
                    break
                
                current_sub_cycles_in_block += 1

                # --- Per-sub-cycle state (Batch-wise independent) ---
                sub_cycle_determined_per_item = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
                history_per_item = [collections.deque(maxlen=cycle_length_stability_window) for _ in range(batch_size)]
                
                block_start_in_gen = block_idx * block_length
                block_end_in_gen = (block_idx + 1) * block_length

                # === PHASE 1: Cycle Length Exploration & Initial Fill ===
                for k_step in range(k_exploration_steps):
                    # --- Model Call ---
                    total_model_calls += 1
                    totol_model_gen_length += gen_length
                    if cfg_scale > 0.0: # Simplified CFG, adapt full logic if needed
                        cfg_x = x.clone()
                        cfg_x[prompt_index_full_x] = mask_id
                        logits_main = model(x, attention_mask=attention_mask).logits
                        cfg_logits_main = model(cfg_x, attention_mask=attention_mask).logits
                        logits_full = logits_main + cfg_scale * (logits_main - cfg_logits_main)
                    else:
                        logits_full = model(x, attention_mask=attention_mask).logits
                    
                    logits_gen_part = logits_full[:, prompt_length:]
                    x0_gen = torch.argmax(add_gumbel_noise(logits_gen_part, temperature), dim=-1)
                    p_gen = F.softmax(logits_gen_part, dim=-1)
                    x0_p_gen = torch.gather(p_gen, dim=-1, index=x0_gen.unsqueeze(-1)).squeeze(-1)
                    

                    current_global_mask_index_gen_part = (x[:, prompt_length:] == mask_id) 
                    confidence_gen_wide = torch.where(current_global_mask_index_gen_part, x0_p_gen, torch.tensor(-np.inf, device=x.device, dtype=x0_p_gen.dtype))

                    # --- Estimate sub-cycle length (focus on current block) ---
                    
                    for b_idx in range(batch_size):
                        if not sub_cycle_determined_per_item[b_idx]:
                            previous_len_item = last_sub_cycle_length_per_item[b_idx].item()

                            observation_abs_start_in_gen = block_start_in_gen + previous_len_item
                            
                            observation_abs_end_in_gen = block_end_in_gen 

                            increment_len = 0 
                            
                            if observation_abs_start_in_gen < observation_abs_end_in_gen:
                                confidence_in_observation_scope = confidence_gen_wide[b_idx, observation_abs_start_in_gen : observation_abs_end_in_gen]
                                
                                if confidence_in_observation_scope.numel() > 0: 
                                    above_thresh_indices_in_scope = (confidence_in_observation_scope >= cycle_len_confidence_threshold).nonzero(as_tuple=True)[0]
                                    
                                    if len(above_thresh_indices_in_scope) > 0:
                                        farthest_idx_in_scope = above_thresh_indices_in_scope.max().item()
                                        increment_len = farthest_idx_in_scope + 1
                                    else:
                                        increment_len = 1 
                                else:
                                    pass 
                            else:
                                increment_len = 0

                            est_len = previous_len_item + increment_len
                            
                            est_len = max(1, est_len) 
                            est_len = min(est_len, block_length) 
                            history_per_item[b_idx].append(est_len)

                            if len(history_per_item[b_idx]) >= cycle_length_stability_window:
                                hist_np = np.array(list(history_per_item[b_idx]))
                                
                                if np.std(hist_np) < cycle_length_stability_std_dev_threshold:
                                    det_len = int(history_per_item[b_idx][-1])
                                    actual_sub_cycle_length_per_item[b_idx] = max(1, min(det_len, block_length))
                                    sub_cycle_determined_per_item[b_idx] = True
                                else:
                                    det_len = int(np.mean(hist_np))
                                    actual_sub_cycle_length_per_item[b_idx] = max(1, min(det_len, block_length))
                                    sub_cycle_determined_per_item[b_idx] = False if k_step < k_exploration_steps - 1 else True
                    # --- Fill some tokens (initial fill for Phase 1) ---
                    num_to_fill_p1 = get_num_tokens_for_phase1_step(mask_in_current_block_abs_coords) 
                    
                    transfer_mask_p1 = torch.zeros_like(x0_gen, dtype=torch.bool) # For full gen_length
                    for b_idx in range(batch_size):
                        previous_len_item_fill = last_sub_cycle_length_per_item[b_idx].item()
                        fill_op_abs_start_in_gen = block_start_in_gen + previous_len_item_fill
                        fill_op_abs_end_in_gen = block_end_in_gen

                        increment_len_p1_fill = 0 

                        if fill_op_abs_start_in_gen < fill_op_abs_end_in_gen:
                            conf_in_fill_op_scope = confidence_gen_wide[b_idx, fill_op_abs_start_in_gen : fill_op_abs_end_in_gen]
                            mask_in_fill_op_scope = (x[b_idx, prompt_length + fill_op_abs_start_in_gen : prompt_length + fill_op_abs_end_in_gen] == mask_id)
                            
                            if conf_in_fill_op_scope.numel() > 0: 
                                eff_conf_in_fill_op_scope = torch.where(mask_in_fill_op_scope, conf_in_fill_op_scope, torch.tensor(-np.inf, device=x.device, dtype=conf_in_fill_op_scope.dtype))
                                
                                num_masked_in_fill_op_scope = mask_in_fill_op_scope.sum().item()
                                
                                if num_to_fill_p1[b_idx] > 0 and num_masked_in_fill_op_scope > 0:
                                    k = min(num_to_fill_p1[b_idx].item(), num_masked_in_fill_op_scope)
                                    phase1_high_conf_fill_indices = (conf_in_fill_op_scope >= high_confidence_threshold) & mask_in_fill_op_scope
                                    if phase1_high_conf_fill_indices.any() and phase1_high_conf_fill_indices.sum().item()>1:
                                        abs_indices_to_fill = fill_op_abs_start_in_gen + phase1_high_conf_fill_indices.nonzero(as_tuple=True)[0]
                                        transfer_mask_p1[b_idx, abs_indices_to_fill] = True
                                    else:           
                                        if k > 0:
                                            top_k_indices_relative_to_fill_scope = torch.topk(eff_conf_in_fill_op_scope, k=k).indices
                                            abs_indices_to_fill_in_gen = fill_op_abs_start_in_gen + top_k_indices_relative_to_fill_scope
                                            transfer_mask_p1[b_idx, abs_indices_to_fill_in_gen] = True
                                
                    x[:, prompt_length:][transfer_mask_p1] = x0_gen[transfer_mask_p1]
                    
                    # Plotting
                    if plot_confidence_maps:
                        title_info_p1 = {
                            "block_idx": block_idx, "sub_cycle": current_sub_cycles_in_block,
                            "phase": "1_Explore", "step_in_phase": k_step,
                            "total_calls": total_model_calls,
                            "end_index": -1
                        }
                        plot_confidence_heatmap(
                            confidence_gen_wide[0],
                            x[0, prompt_length:],
                            mask_id, title_info_p1, plot_save_dir, gen_length
                        )
                    
                # After k_exploration_steps, if any item's sub-cycle length is not determined, use a fallback.
                for b_idx in range(batch_size):
                    if not sub_cycle_determined_per_item[b_idx]:
                        if len(history_per_item[b_idx]) > 0: # Use average of what was gathered
                            actual_sub_cycle_length_per_item[b_idx] = max(1, min(int(np.mean(list(history_per_item[b_idx]))), block_length))
                        else: # Absolute fallback
                            actual_sub_cycle_length_per_item[b_idx] = block_length // 2 # Or some other default
                        sub_cycle_determined_per_item[b_idx] = True # Mark as determined for next phases


                # === PHASE 2&3: High-Confidence Fill ===
                phase_2_and_3_calls = 0
                # cache list
                cache_out_cycle_logits_list = []
                cache_out_cycle_cfg_logits_list = []
                cache_out_cycle_full_logits_list = []
                # cycle list
                active_region_start_check_list = []
                active_region_end_check_list = []
                while True:
                    all_p2_active_regions_filled_for_all_items = True
                    for b_idx_check in range(batch_size):
                        current_cumulative_len_check = actual_sub_cycle_length_per_item[b_idx_check].item()
                        previous_cumulative_len_check = last_sub_cycle_length_per_item[b_idx_check].item()
                        active_region_start_check_list.append(block_start_in_gen + previous_cumulative_len_check)
                        active_region_end_check_list.append(block_start_in_gen + current_cumulative_len_check)
                        if active_region_start_check_list[b_idx_check] < active_region_end_check_list[b_idx_check]:
                            mask_in_ar_check = (x[b_idx_check, prompt_length + active_region_start_check_list[b_idx_check] : prompt_length + active_region_end_check_list[b_idx_check]] == mask_id)
                            if mask_in_ar_check.any(): # If any mask exists in this item's active region
                                all_p2_active_regions_filled_for_all_items = False
                                break # No need to check other items, we know P2 still has work
                    if all_p2_active_regions_filled_for_all_items:
                        break 
                    
                    phase_2_and_3_calls += 1
                    total_model_calls += 1
                    
                    # model call
                    if cfg_scale > 0.0: # Simplified CFG
                        if phase_2_and_3_calls == 1:
                            cfg_x = x.clone()
                            cfg_x[prompt_index_full_x] = mask_id
                            logits_main = model(x, attention_mask=attention_mask).logits 
                            cfg_logits_main = model(cfg_x, attention_mask=attention_mask).logits
                            for b_idx_check in range(batch_size):
                                cache_out_cycle_logits_list.append(logits_main[b_idx_check,  prompt_length + active_region_end_check_list[b_idx_check]:].unsqueeze(0))
                                cache_out_cycle_cfg_logits_list.append(cfg_logits_main[b_idx_check,  prompt_length + active_region_end_check_list[b_idx_check]:].unsqueeze(0))
                            logits_full = logits_main + cfg_scale * (logits_main - cfg_logits_main)
                        else:
                            cfg_x = x.clone()
                            cfg_x[prompt_index_full_x] = mask_id
                            logits_main_batch = []
                            cfg_logits_main_batch = []
                            for b_idx_check in range(batch_size):
                                logits_main_part = model(x[:, :prompt_length + active_region_end_check_list[b_idx_check]], attention_mask=attention_mask).logits
                                cfg_logits_main_part = model(cfg_x[:, :prompt_length + active_region_end_check_list[b_idx_check]], attention_mask=attention_mask).logits
                                logits_main_batch.append(torch.cat([logits_main_part[b_idx_check].unsqueeze(0), cache_out_cycle_logits_list[b_idx_check]], dim=1))
                                cfg_logits_main_batch.append(torch.cat([cfg_logits_main_part[b_idx_check].unsqueeze(0), cache_out_cycle_cfg_logits_list[b_idx_check]],dim=1))
                            logits_main = torch.cat(logits_main_batch, dim=0)
                            cfg_logits_main = torch.cat(cfg_logits_main_batch, dim=0)
                            logits_full = logits_main + cfg_scale * (logits_main - cfg_logits_main)
                    else:
                        if phase_2_and_3_calls == 1:
                            # logits_full [1, 1149, 126464]
                            totol_model_gen_length += gen_length
                            logits_full = model(x, attention_mask=attention_mask).logits
                            for b_idx_check in range(batch_size):
                                cache_out_cycle_full_logits_list.append(logits_full[b_idx_check, prompt_length + active_region_end_check_list[b_idx_check]:].unsqueeze(0))
                        else:
                            logits_full_batch = []
                            for b_idx_check in range(batch_size):
                                logits_full_item = model(x[:, :prompt_length + active_region_end_check_list[b_idx_check]], attention_mask=attention_mask).logits
                                logits_full_batch.append(torch.cat([logits_full_item[b_idx_check].unsqueeze(0), cache_out_cycle_full_logits_list[b_idx_check]], dim=1))
                                # print(f"{logits_full_batch.shape}")
                                totol_model_gen_length += active_region_end_check_list[b_idx_check]

                            logits_full = torch.cat(logits_full_batch, dim=0)
                        
                    
                    logits_gen_part = logits_full[:, prompt_length:]
                    x0_gen = torch.argmax(add_gumbel_noise(logits_gen_part, temperature), dim=-1)
                    p_gen = F.softmax(logits_gen_part, dim=-1)
                    x0_p_gen = torch.gather(p_gen, dim=-1, index=x0_gen.unsqueeze(-1)).squeeze(-1)
                    current_global_mask_index_gen_part = (x[:, prompt_length:] == mask_id)
                    confidence_gen_wide = torch.where(current_global_mask_index_gen_part, x0_p_gen, torch.tensor(-np.inf, device=x.device, dtype=x0_p_gen.dtype))
                    transfer_mask_p2_and_p3 = torch.zeros_like(x0_gen, dtype=torch.bool)
                    
                    for b_idx in range(batch_size):
                        sub_cycle_abs_end_in_gen = block_start_in_gen + actual_sub_cycle_length_per_item[b_idx].item()
                        sub_cycle_abs_start_in_gen = block_start_in_gen + last_sub_cycle_length_per_item[b_idx].item() 
                        

                        conf_in_sub_cycle_scope = confidence_gen_wide[b_idx, sub_cycle_abs_start_in_gen:sub_cycle_abs_end_in_gen]
                        mask_in_sub_cycle_scope = (x[b_idx, prompt_length + sub_cycle_abs_start_in_gen : prompt_length + sub_cycle_abs_end_in_gen] == mask_id)

                        high_conf_fill_indices = (conf_in_sub_cycle_scope >= high_confidence_threshold) & mask_in_sub_cycle_scope

                        # print(f"high_conf_fill_indices{high_conf_fill_indices}")
                        
                        if high_conf_fill_indices.any() and high_conf_fill_indices.sum().item()>1:
                            abs_indices_to_fill = sub_cycle_abs_start_in_gen + high_conf_fill_indices.nonzero(as_tuple=True)[0]
                            transfer_mask_p2_and_p3[b_idx, abs_indices_to_fill] = True
                        else:
                            n2_num_transfer_tokens = get_num_tokens_for_phase3_step(mask_in_current_block_abs_coords)
                            eff_conf_sub_cycle = torch.where(mask_in_sub_cycle_scope, conf_in_sub_cycle_scope, torch.tensor(-np.inf, device=x.device, dtype=conf_in_sub_cycle_scope.dtype))
                    
                            top_k_indices_relative_to_sub_cycle = torch.topk(eff_conf_sub_cycle, k=n2_num_transfer_tokens[b_idx].item()).indices
                            abs_indices_to_fill = sub_cycle_abs_start_in_gen + top_k_indices_relative_to_sub_cycle
                            transfer_mask_p2_and_p3[b_idx, abs_indices_to_fill] = True
                            
                    x[:, prompt_length:][transfer_mask_p2_and_p3] = x0_gen[transfer_mask_p2_and_p3] # Update x
                    
                    # --- MODIFIED: Plotting for Phase 2&3 ---
                    if plot_confidence_maps:
                        title_info_p2 = {
                            "block_idx": block_idx, "sub_cycle": current_sub_cycles_in_block,
                            "phase": "2&3_HighConf&fix", "step_in_phase": phase_2_and_3_calls,
                            "total_calls": total_model_calls,
                            "end_index":block_start_in_gen + actual_sub_cycle_length_per_item[0].item()
                        }
                        plot_confidence_heatmap(
                            confidence_gen_wide[0],
                            x[0, prompt_length:],
                            mask_id, title_info_p2, plot_save_dir, gen_length
                        )
            
                last_sub_cycle_length_per_item = actual_sub_cycle_length_per_item.clone()
                
        # print(f"\nGeneration complete. Total model calls (generate step): {total_model_calls}")
        avg_model_gen_length = totol_model_gen_length / total_model_calls

        return x[:, prompt_length:], total_model_calls, avg_model_gen_length
        # return x[:, prompt_length:]