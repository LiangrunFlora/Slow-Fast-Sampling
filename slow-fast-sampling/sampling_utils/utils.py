import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel, 
    RowwiseParallel, 
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def apply_tensor_parallel(model, tp_mesh):
    # 为 LLaDA 模型定义 TP 并行计划
    model_tp_plan = {
        "transformer.wte": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
        "transformer.ln_f": SequenceParallel(),
        "transformer.ff_out": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate())
    }
    
    block_tp_plan = {
        "attn_norm": SequenceParallel(),
        "q_proj": ColwiseParallel(),
        "k_proj": ColwiseParallel(),
        "v_proj": ColwiseParallel(),
        "attn_out": RowwiseParallel(output_layouts=Shard(1)),
        "ff_norm": SequenceParallel(),
        "ff_proj": ColwiseParallel(),
        "up_proj": ColwiseParallel(),
        "ff_out": RowwiseParallel(output_layouts=Shard(1)),
        "self": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),)
        )
    }
    model.transformer.blocks[0].config.n_heads = model.transformer.blocks[0].config.n_heads // tp_mesh.size()
    model.transformer.blocks[0].config.n_kv_heads = model.transformer.blocks[0].config.n_kv_heads // tp_mesh.size()
    for block in model.transformer.blocks:
        
        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=block_tp_plan
        )

    # 对整体模型应用 TP
    model = parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        parallelize_plan=model_tp_plan
    )
    
    return model

