# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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

import torch
from typing import Optional, Tuple, List, Union
from torch.nn.functional import scaled_dot_product_attention
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import (
    logger,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from unsloth.kernels import *
from unsloth.models._utils import *
from unsloth.models._utils import __version__
if HAS_FLASH_ATTENTION:
    from flash_attn import flash_attn_func

# Final patching code
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
) 
from peft import PeftModelForCausalLM
import gc
import peft
import bitsandbytes as bnb
import numpy as np
import types

from unsloth.models.llama import *
from .modeling_rankllama import LlamaForRankCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from transformers import set_seed as transformers_set_seed
from peft import LoraConfig, TaskType, get_peft_model as _get_peft_model


def fast_forward_outputs(
    self,
    input_ids: torch.LongTensor = None,
    causal_mask: Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    *args, **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:

    if causal_mask is None:
        causal_mask = xformers.attn_bias.LowerTriangularMask()

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        causal_mask=causal_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    return outputs



def fast_sft_loss(
    self,
    outputs: CausalLMOutputWithPast,
    labels: Optional[torch.LongTensor] = None,
    classes: Optional[torch.Tensor] = None,
    *args, **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    # if labels is None or (classes is not None and sum(classes[:, 1]) == 0):
    if labels is None:
        return CausalLMOutputWithPast(
            loss=torch.tensor(0, device=logits.device),
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    if labels is not None:
        # if classes is not None:
        #     classes = classes.to(torch.bool)
        #     logits = logits[classes[:, 1]].contiguous()
        #     labels = labels[classes[:, 1]].contiguous()
        shift_logits = logits
        if not hasattr(self, "extra_ignored_labels"):
            # Fixes https://github.com/unslothai/unsloth/issues/10
            self.extra_ignored_labels = torch.full((self.max_seq_length, 1), -100, device = "cuda")
        pass
        
        shift_labels = torch.hstack((labels[..., 1:], self.extra_ignored_labels[:labels.shape[0]]))
        loss = fast_cross_entropy_loss(
            logits = shift_logits,
            labels = shift_labels,
        )

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
pass


def PeftModelForCausalLM_fast_forward(
    self,
    input_ids=None,
    causal_mask=None,
    attention_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    task_ids=None,
    **kwargs,
):
    return self.base_model(
        input_ids=input_ids,
        causal_mask=causal_mask,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        labels=labels,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **kwargs,
    )
pass


class FastRankLlamaModel(FastLlamaModel):

    @staticmethod
    def pre_patch():
        LlamaAttention      .forward = LlamaAttention_fast_forward
        LlamaDecoderLayer   .forward = LlamaDecoderLayer_fast_forward
        LlamaModel          .forward = LlamaModel_fast_forward
        LlamaForRankCausalLM.forward_outputs = fast_forward_outputs
        LlamaForRankCausalLM.sft_loss = fast_sft_loss
        PeftModelForCausalLM.forward = PeftModelForCausalLM_fast_forward
        return
    pass


    @staticmethod
    def from_pretrained(
        model_name,
        config,
        max_seq_length = 4096,
        torch_dtype = None,
        resize_token_embeddings = None,
        **kwargs,
    ):
        SUPPORTS_BFLOAT16 = torch.cuda.is_bf16_supported()
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        statistics = \
           f"==((====))==  Unsloth: Fast Llama patching release {__version__}\n"\
           f"   \\\   /|    GPU: {gpu_stats.name}. Max memory: {max_memory} GB\n"\
           f"O^O/ \_/ \\    CUDA capability = {gpu_stats.major}.{gpu_stats.minor}. Xformers = {xformers_version}. FA = {HAS_FLASH_ATTENTION}.\n"\
           f"\        /    Pytorch version: {torch.__version__}. CUDA Toolkit = {torch.version.cuda}\n"\
           f' "-____-"     bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. Platform = {platform_system}\n'
        logger.warning_once(statistics)
        FastRankLlamaModel.pre_patch()

        if torch_dtype is None:
            torch_dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif torch_dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            logger.warning_once("Device does not support bfloat16. Will change to float16.")
            torch_dtype = torch.float16

        assert(torch_dtype == torch.float16 or torch_dtype == torch.bfloat16 or torch_dtype == torch.float32)

        model = LlamaForRankCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype = torch_dtype,
            **kwargs
        )
        if resize_token_embeddings:
            model.resize_token_embeddings(resize_token_embeddings, pad_to_multiple_of=8)

        model = FastLlamaModel.post_patch(model)

        # Patch up QKV / O and MLP
        for idx, layer in enumerate(model.model.layers):
            layer.self_attn.apply_qkv = original_apply_qkv
            layer.self_attn.apply_o   = original_apply_o
        pass

        model.max_seq_length = max_seq_length
        model.model.max_seq_length = max_seq_length
        return model
    pass


    @staticmethod
    def post_patch(model):
        # Patch model
        layers = model.model.layers

        # Torch.compile fails on embedding matrix??
        # Workaround randomnly fixes it for torch versions < 2.2
        model.model.embed_tokens = torch.nn.Embedding.from_pretrained(model.model.embed_tokens.weight)
        model.config.update({"unsloth_version" : __version__})

        # We also do this for the lm_head
        lm_head = torch.nn.Linear(1, 1, bias = None)
        del lm_head.weight
        lm_head.weight = model.lm_head.weight
        lm_head.in_features  = lm_head.weight.shape[1]
        lm_head.out_features = lm_head.weight.shape[0]
        model.lm_head = lm_head

        # Also patch all dtypes - BnB seems to not allocate the correct type?
        # BnB default dtype seems to be float16!
        correct_dtype = lm_head.weight.dtype

        for name, module in model.named_modules():
            if isinstance(module, (bnb.nn.Linear4bit, peft.tuners.lora.Linear4bit)):
                weight = module.weight
                quant_state = weight.quant_state

                if type(quant_state) is list:
                    # BnB seems to have float16 as default!
                    module.weight.quant_state[2] = correct_dtype # Cast to correct dtype
                else:
                    # https://github.com/TimDettmers/bitsandbytes/pull/763/files
                    quant_state.dtype = correct_dtype
                pass
            pass
        pass

        # Clear deleted GPU items
        gc.collect()
        torch.cuda.empty_cache()
        return model
    pass


    @staticmethod
    def get_peft_model(
        model,
        max_seq_length=4096,
        lora_config=None,
        use_gradient_checkpointing = True,
        random_state = 42,
    ):

        transformers_set_seed(random_state)
        model.config.update({"unsloth_version" : __version__})

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing = use_gradient_checkpointing,
            use_reentrant = True,
        )
        model = _get_peft_model(model, lora_config)

        # Do patching
        n_mlp = 0
        n_qkv = 0
        n_o   = 0
        for idx, layer in enumerate(model.model.model.layers):

            # MLP patching
            if  hasattr(layer.mlp.gate_proj, "lora_A") and \
                hasattr(layer.mlp.  up_proj, "lora_A") and \
                hasattr(layer.mlp.down_proj, "lora_A"):

                # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
                layer.mlp.forward = types.MethodType(apply_lora_mlp, layer.mlp)
                n_mlp += 1
            pass

            # QKV attention patching
            if  hasattr(layer.self_attn.q_proj, "lora_A") and \
                hasattr(layer.self_attn.k_proj, "lora_A") and \
                hasattr(layer.self_attn.v_proj, "lora_A"):

                layer.self_attn.apply_qkv = apply_lora_qkv
                n_qkv += 1
            pass

            # O attention patching
            if hasattr(layer.self_attn.o_proj, "lora_A"):

                layer.self_attn.apply_o = apply_lora_o
                n_o += 1
            pass
        pass

        logger.warning_once(
            f"Unsloth {__version__} patched {len(model.model.model.layers)} layers with "\
            f"{n_qkv} QKV layers, {n_o} O layers and {n_mlp} MLP layers.",
        )

        # Patch cross entropy loss labels
        # Fixes https://github.com/unslothai/unsloth/issues/10
        extra_ignored_labels = torch.full((max_seq_length, 1), -100, device = "cuda")
        model.model.extra_ignored_labels = extra_ignored_labels
        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model.max_seq_length = max_seq_length
            internal_model = internal_model.model
        pass
        internal_model.max_seq_length = max_seq_length
        return model
    pass
pass
