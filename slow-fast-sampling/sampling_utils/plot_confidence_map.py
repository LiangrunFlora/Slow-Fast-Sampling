import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch 
import numpy as np 


def plot_confidence_heatmap(
    confidence_tensor_item,  # 单个batch item的置信度张量 (gen_length,)
    x_tensor_item_gen_part,  # 单个batch item的x张量的生成部分 (gen_length,)
    mask_id,
    title_info,              # 包含 block, sub-cycle, phase, step 等信息的字典
    save_dir="confidence_plots",
    gen_length=128 # 需要知道gen_length以正确显示x轴
):
    """绘制并保存单个batch item的置信度热力图。"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    conf_np = confidence_tensor_item.to(torch.float32).cpu().numpy()
    x_np_gen = x_tensor_item_gen_part.cpu().numpy()

    # 将 -np.inf 替换为 NaN 以便绘图，但保留原始值用于注释
    conf_for_heatmap = conf_np.copy()
    conf_for_heatmap[np.isinf(conf_for_heatmap)] = np.nan

    # 创建注释文本：显示置信度，如果不是mask则显示token ID
    annot_text = np.full_like(conf_np, "", dtype=object).reshape(1, -1)
    for i in range(gen_length):
        if x_np_gen[i] != mask_id:
            annot_text[0, i] = f"T:{x_np_gen[i]}" # 已填充的token
        elif not np.isnan(conf_for_heatmap[i]): # 是mask且有置信度
             annot_text[0, i] = f"{conf_np[i]:.2f}"
        # else: # 是mask且置信度为-inf (或NaN)
        #    annot_text[0, i] = "Mask" # 或保持为空


    plt.figure(figsize=(max(15, gen_length // 5), 5))
    sns.heatmap(
        conf_for_heatmap.reshape(1, -1),
        annot=annot_text, # 使用自定义注释
        fmt="s", # 使用字符串格式的注释 (因为annot_text是字符串)
        cmap="viridis",
        cbar=True,
        xticklabels=gen_length // 10 if gen_length > 20 else True,
        yticklabels=False,
        linewidths=.5,
        linecolor='gray',
        # vmin=0, vmax=1 # 如果置信度是概率的话
    )
    
    title = (f"Conf: Blk {title_info['block_idx']}, SubCyc {title_info['sub_cycle']}, "
             f"Ph {title_info['phase']}, Step {title_info['step_in_phase']} "
             f"(TotalCalls {title_info['total_calls']})")
    plt.title(title, fontsize=10)
    plt.xlabel("Token Position in Generation")
    plt.tight_layout()

    filename = (f"conf_blk{title_info['block_idx']}_sc{title_info['sub_cycle']}_"
                f"ph{title_info['phase']}_st{title_info['step_in_phase']}_"
                f"tc{title_info['total_calls']}_endindex{title_info['end_index']}.png")
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()