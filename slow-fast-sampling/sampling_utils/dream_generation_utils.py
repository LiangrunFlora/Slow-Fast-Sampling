# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)
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

logger = logging.get_logger(__name__)


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None



class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )
        avg_model_calls_length = 0
        global_model_calls = 0
        
        # result = self._sample(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     generation_config=generation_config,
        #     generation_tokens_hook_func=generation_tokens_hook_func,
        #     generation_logits_hook_func=generation_logits_hook_func
        # )
        result, avg_model_calls_length, global_model_calls = self._slow_fast_sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func
        )
        return result, avg_model_calls_length, global_model_calls

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        
        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        
        for i in range(steps):
            mask_index = (x == mask_token_id)
            logits = self(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
        
            if alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            else:
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                num_mask_token = mask_index.sum()
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                    else:
                        confidence = confidence / alg_temp
                        confidence = F.softmax(confidence, dim=-1)
                        transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                    x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_token_id
                    x0_[transfer_index] = x0[transfer_index].clone()
                    x[mask_index] = x0_

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())
        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x
        
        
        
    def _slow_fast_sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor], # This is the original attention mask
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        plot_confidence_maps = False
        plot_save_dir = 'confidence_dream_map'
        total_model_calls_length = 0

        # 从 generation_config 获取参数
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        
        # max_iterations is the total budget for the entire generation process
        max_total_iterations = generation_config.steps 
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        # 大循环的次数 (新超参数)
        MAX_MAJOR_CYCLES = getattr(generation_config, "max_major_cycles", 5)

        # 阶段0 (周期发现) 的参数
        N_CYCLE_DISCOVERY_STEPS = getattr(generation_config, "n_cycle_discovery_steps", 6) # 每个大循环内，发现阶段的步数
        CONF_THRESHOLD_CYCLE_LEN_PRED = getattr(generation_config, "conf_threshold_cycle_len_pred", 0.1)
        STABILIZE_COUNT = getattr(generation_config, "stabilize_count", 2) # 建议 >=2
        MIN_CYCLE_LEN_FOR_STAB = getattr(generation_config, "min_cycle_len_for_stab", 3)
        CYCLE_LEN_VARIANCE_THRESHOLD = getattr(generation_config, "cycle_len_variance_threshold", 1.0)
        HIGH_CONF_FOR_PARALLEL_DECODE_DISCOVERY = getattr(generation_config, "high_conf_for_parallel_decode_discovery", 0.8)
        MIN_TOKENS_FOR_PARALLEL_DECODE_DISCOVERY = getattr(generation_config, "min_tokens_for_parallel_decode_discovery", 2)

        # 阶段1 (周期解码) 的参数
        # MAX_STEPS_PER_CYCLE_DECODE = getattr(generation_config, "max_steps_per_cycle_decode", 10) # 每个大循环内，阶段1的最大步数
        HIGH_CONF_FOR_CYCLE_DECODE = getattr(generation_config, "high_conf_for_cycle_decode", 0.9)
        MIN_DECODE_IN_CYCLE_FALLBACK = getattr(generation_config, "min_decode_in_cycle_fallback", 2)
        
        histories = [] if (return_dict_in_generate and output_history) else None

        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        batch_size = x.shape[0]
        device = x.device
        prompt_length = input_ids.shape[1]
        gen_length = max_length - input_ids.shape[1]

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask_padded = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask_padded.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask_padded == 0, 1)
            attention_mask_for_model = torch.logical_and(
                attention_mask_padded.unsqueeze(1).unsqueeze(-2),
                attention_mask_padded.unsqueeze(1).unsqueeze(-1),
            )
        else:
            print(f"enter")
            tok_idx = None
            attention_mask_for_model = "full"

        x = generation_tokens_hook_func(None, x, None) # Initial hook

        # --- 状态变量 ---
        stable_cycle_length_batch = torch.full((batch_size, max_length), 0, device=device, dtype=torch.long) # 初始化一个默认值
        is_cycle_len_determined_batch = torch.zeros(batch_size, device=device, dtype=torch.bool) # Reset for each major cycle's phase 0
        potential_cycle_lengths_history_tensor = torch.zeros((STABILIZE_COUNT, batch_size), device=device, dtype=torch.long)
        
        global_model_calls = 0 # Tracks total calls to the model (like global_iter_step before)

        # ============================================================
        #  OUTER MAJOR LOOP
        # ============================================================
        major_cycle_iter = -1
        while True:
            major_cycle_iter += 1
            if not (x == mask_token_id).any():
                break
            # Reset determination status for phase 0 of this major cycle
            is_cycle_len_determined_batch.fill_(False)
            potential_cycle_lengths_history_tensor.fill_(0)

            # ============================================================
            #  PHASE 0: CYCLE DISCOVERY (within major_cycle_iter)
            # ============================================================
            # logger.info(f"  大循环 {major_cycle_iter+1} - 进入周期发现阶段...")
            for discovery_step in range(N_CYCLE_DISCOVERY_STEPS):
                if not (x == mask_token_id).any(): 
                    break
                
                mask_positions_bool_p0 = (x == mask_token_id)
                
                model_outputs_p0 = self(x, attention_mask_for_model, tok_idx)
                logits_p0 = model_outputs_p0.logits
                logits_p0 = torch.cat([logits_p0[:,:1], logits_p0[:, :-1]], dim=1)
                logits_p0 = generation_logits_hook_func(global_model_calls, x, logits_p0)
                
                total_model_calls_length += gen_length

                x_updated_p0 = x.clone()
                confidence = None
                x0 = None

                for b_idx in range(batch_size):
                    # 获取全局的prompt之后的confidence和x0
                    probs = torch.softmax(logits_p0[b_idx, prompt_length:], dim=-1)
                    confidence, x0 = probs.max(dim=-1)

                    indices_to_decode_p0 = torch.tensor([], dtype=torch.long, device=device)

                    start_abs_idx = prompt_length + stable_cycle_length_batch[b_idx, major_cycle_iter-1]
                    start_cycle_idx = stable_cycle_length_batch[b_idx, major_cycle_iter-1]
                    
                    confidence_undecoded_cycle = torch.where(mask_positions_bool_p0[b_idx, start_abs_idx:], confidence[start_cycle_idx:], torch.tensor(-np.inf, device=x.device, dtype=confidence.dtype))
                    if not is_cycle_len_determined_batch[b_idx]:
                        above_thresh_indices_in_scope = (confidence_undecoded_cycle >= CONF_THRESHOLD_CYCLE_LEN_PRED).nonzero(as_tuple=True)[0]

                        if len(above_thresh_indices_in_scope) > 0:
                            farthest_idx_in_scope = above_thresh_indices_in_scope.max().item()
                            potential_len = start_cycle_idx + farthest_idx_in_scope + 1 # 增量长度,末尾idx
                        else:
                            potential_len = start_cycle_idx + 1 # 尝试至少扩展1 (如果后面还有空间)
                        current_history_slot = discovery_step % STABILIZE_COUNT
                        potential_cycle_lengths_history_tensor[current_history_slot, b_idx] = potential_len
                        
                        if discovery_step >= 0:
                            # 提取当前批次项的周期长度历史窗口
                            history_window_for_item_list = [
                                potential_cycle_lengths_history_tensor[i, b_idx].item() for i in range(STABILIZE_COUNT)
                            ]
                            history_window_tensor = torch.tensor(history_window_for_item_list, 
                                                                device=x.device, dtype=torch.float32)
                            # 计算方差
                            variance_of_lengths = torch.tensor(float('inf'), device=x.device) # 默认值，表示不稳定
                            if STABILIZE_COUNT >= 2: # 方差对于 unbiased=True 至少需要2个点
                                variance_of_lengths = torch.var(history_window_tensor, unbiased=True)
                            elif STABILIZE_COUNT == 1: # 如果只看一步历史，方差视为0 (最稳定)
                                variance_of_lengths = torch.tensor(0.0, device=x.device)
                            is_variance_low = variance_of_lengths <= CYCLE_LEN_VARIANCE_THRESHOLD

                            if is_variance_low:
                                # 方差小：将稳定周期长度设置为历史窗口中的最后一个值 (即 potential_len)
                                stable_cycle_length_batch[b_idx, major_cycle_iter] = potential_len
                                is_cycle_len_determined_batch[b_idx] = True
                            else:
                                # 方差大：将稳定周期长度设置为历史窗口中长度的平均值
                                mean_len = torch.mean(history_window_tensor)
                                stable_cycle_length_batch[b_idx, major_cycle_iter] = torch.round(mean_len).long().clamp(min=1) # 四舍五入并确保至少为1
                                is_cycle_len_determined_batch[b_idx] = False if discovery_step < N_CYCLE_DISCOVERY_STEPS -1 else True# 明确标记为不稳定
                                
                    # decode
                    num_high_conf = (confidence_undecoded_cycle >= HIGH_CONF_FOR_PARALLEL_DECODE_DISCOVERY).sum().item()
                    if num_high_conf >= MIN_TOKENS_FOR_PARALLEL_DECODE_DISCOVERY:
                        indices_to_decode_p0 = torch.where(confidence_undecoded_cycle >= HIGH_CONF_FOR_PARALLEL_DECODE_DISCOVERY)[0]
                    elif len(confidence_undecoded_cycle) > 0:
                        _, indices_to_decode_p0 = torch.topk(confidence_undecoded_cycle, k=1)
                    
                    if len(indices_to_decode_p0) > 0:
                        orig_indices = indices_to_decode_p0 + start_abs_idx
                        tokens_fill = x0[indices_to_decode_p0 + start_cycle_idx]
                        x_updated_p0[b_idx, orig_indices] = tokens_fill
                        
                
                x = x_updated_p0
                x = generation_tokens_hook_func(global_model_calls, x, logits_p0)
                if histories is not None: histories.append(x.clone())
                global_model_calls += 1

                
                if plot_confidence_maps:
                        title_info_p2 = {
                            "block_idx":0, "sub_cycle": major_cycle_iter,
                            "phase": "1", "step_in_phase": N_CYCLE_DISCOVERY_STEPS,
                            "total_calls": global_model_calls,
                            "end_index":-1
                        }
                        confidence_gen_wide = torch.where(mask_positions_bool_p0[b_idx, prompt_length:], confidence, torch.tensor(-np.inf, device=x.device, dtype=confidence.dtype))
                        plot_confidence_heatmap(
                            confidence_gen_wide,
                            x[0, prompt_length:],
                            151666, title_info_p2, plot_save_dir, max_length - prompt_length
                        )

            # ============================================================
            #  PHASE 1: CYCLE DECODE (within major_cycle_iter)
            # ============================================================
            # logger.info(f"  大循环 {major_cycle_iter+1} - 进入周期解码阶段...")

            phase1_steps_this_major_cycle = 0
            # Heuristic: If a few steps in phase 1 yield no decoded tokens for any active item, break this phase1.
            NO_PROGRESS_PATIENCE_P1 = 3 
            no_progress_counter_p1 = torch.zeros(batch_size, dtype=torch.int, device=device)
            cache_for_out_cycle = None
            is_finish = False

            while True: # Loop for Phase 1 steps within this major_cycle
                start_cycle_idx = stable_cycle_length_batch[b_idx, major_cycle_iter-1]
                end_cycle_idx = stable_cycle_length_batch[b_idx, major_cycle_iter]
                start_abs_idx = stable_cycle_length_batch[b_idx, major_cycle_iter-1] + prompt_length
                end_abs_idx = stable_cycle_length_batch[b_idx, major_cycle_iter] + prompt_length

                mask_positions_bool_p1 = (x == mask_token_id)
                cycle_mask_p1 = mask_positions_bool_p1[b_idx, start_abs_idx:end_abs_idx]

                if not cycle_mask_p1.any(): 
                    is_finish = True
                    break

                any_progress_this_p1_step = False # Flag to see if any token was decoded in this p1 step across batch
                if phase1_steps_this_major_cycle == 0:
                    logits_p1 = self(x, attention_mask_for_model, tok_idx).logits
                    cache_for_out_cycle = logits_p1[:, end_abs_idx:]
                    total_model_calls_length += gen_length
                else:
                    current_tok_idx_p1 = tok_idx[:, :end_abs_idx] if tok_idx is not None else None
                    current_attention_mask_p1 = attention_mask_for_model[:, :, :end_abs_idx, :end_abs_idx] \
                                            if attention_mask_for_model != "full" and attention_mask_for_model is not None \
                                            else attention_mask_for_model 
                    logits_p1 = self(x[:, :end_abs_idx], current_attention_mask_p1, current_tok_idx_p1).logits
                    logits_p1 = torch.cat([logits_p1, cache_for_out_cycle], dim=1)
                    total_model_calls_length += end_cycle_idx
                
                # logits_p1 = model_outputs_p1.logits
                logits_p1 = torch.cat([logits_p1[:,:1], logits_p1[:, :-1]], dim=1)
                logits_p1 = generation_logits_hook_func(global_model_calls, x, logits_p1)

                x_updated_p1 = x.clone()
                confidence_p1 = None
                x0_p1 = None

                for b_idx in range(batch_size):
                    if (x[b_idx] == mask_token_id).sum() == 0:
                        no_progress_counter_p1[b_idx] = 0 # Reset patience if done
                        continue
                    
                    # 获取全局的confidence和x0(prompt之后的)
                    probs_p1 = torch.softmax(logits_p1[b_idx, prompt_length:], dim=-1)
                    confidence_p1, x0_p1 = probs_p1.max(dim=-1)

                    confidence_undecoded_in_cycle = torch.where(mask_positions_bool_p1[b_idx, start_abs_idx:end_abs_idx], confidence_p1[start_cycle_idx:end_cycle_idx], torch.tensor(-np.inf, device=x.device, dtype=confidence_p1.dtype))
                    if start_abs_idx < end_abs_idx:
                        very_high_conf_idx = (confidence_undecoded_in_cycle >= HIGH_CONF_FOR_CYCLE_DECODE).nonzero(as_tuple=True)[0]

                        if len(very_high_conf_idx) < MIN_DECODE_IN_CYCLE_FALLBACK:
                            _, indices_to_decode_p1 = torch.topk(confidence_undecoded_in_cycle, k=1)
                        else:
                            indices_to_decode_p1 = very_high_conf_idx 

                    
                    if len(indices_to_decode_p1) > 0:
                        orig_indices = indices_to_decode_p1 + start_abs_idx
                        tokens_fill = x0_p1[indices_to_decode_p1 + start_cycle_idx]
                        x_updated_p1[b_idx, orig_indices] = tokens_fill
                        any_progress_this_p1_step = True
                        no_progress_counter_p1[b_idx] = 0 # Progress made for this item
                    else:
                        no_progress_counter_p1[b_idx] +=1

                x = x_updated_p1
                
                x = generation_tokens_hook_func(global_model_calls, x, logits_p1)
                if histories is not None: histories.append(x.clone())
                global_model_calls += 1
                phase1_steps_this_major_cycle +=1

            
                if plot_confidence_maps:
                        title_info_p2 = {
                            "block_idx":0, "sub_cycle": major_cycle_iter,
                            "phase": "2&3_HighConf&fix", "step_in_phase": phase1_steps_this_major_cycle,
                            "total_calls": global_model_calls,
                            "end_index":end_cycle_idx
                        }
                        confidence_gen_wide = torch.where(mask_positions_bool_p1[b_idx, prompt_length:], confidence_p1, torch.tensor(-np.inf, device=x.device, dtype=confidence_p1.dtype))
                        plot_confidence_heatmap(
                            confidence_gen_wide,
                            x[0, prompt_length:],
                            151666, title_info_p2, plot_save_dir, max_length - prompt_length
                        )

        
        # --- End of OUTER MAJOR LOOP ---
        avg_model_calls_length = total_model_calls_length / global_model_calls

        final_num_masks_left = (x == mask_token_id).sum()
        if final_num_masks_left > 0:
             if global_model_calls >= max_total_iterations:
                 logger.warning(f"达到总模型调用上限 ({max_total_iterations}), 但仍有 {final_num_masks_left.item()} MASK token 未填充.")
             else:
                 logger.warning(f"大循环结束, 但仍有 {final_num_masks_left.item()} MASK token 未填充.")

        if return_dict_in_generate:
            return DreamModelOutput(sequences=x, history=histories), avg_model_calls_length, global_model_calls
        else:
            return x, avg_model_calls_length, global_model_calls