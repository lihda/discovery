from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
from torch import nn
from typing import *
class LlamaForRankCausalLM(LlamaForCausalLM):

    def __init__(
        self, 
        config: "LlamaConfig", 
        rel_score_id: int = 32002, 
        verify_score_id: int = 2, 
        beta: float = 0.5, 
        rr_loss_fn : str = "bce", 
        num_labels: int = 1, 
        enable_verify: bool = False, 
        bias: float = 0.4, 
        smoothing: bool = False,
        psg_num: int = 4, 
        rank_only: bool = False,
        relevant_init: Union[bool, str, int] = None,
        ):
        super().__init__(config)

        self.num_labels = num_labels
        self.beta = beta
        
        self.rel_score = nn.Linear(config.hidden_size, num_labels, bias=False)
        self.rel_score_id = rel_score_id

        self.list_wise = False
        self.rank_only = rank_only
        self.enable_verify = enable_verify
        
        if relevant_init:
            if isinstance(relevant_init, str):
                if relevant_init == "relevant":
                    weight_1 = self.lm_head.weight[8018: 8019].detach().clone()
                    weight_2 = self.lm_head.weight[28190: 28191].detach().clone()
                elif relevant_init == "useful":
                    weight_1 = self.lm_head.weight[5407: 5408].detach().clone()
                    weight_2 = self.lm_head.weight[19315: 19316].detach().clone()
                else:
                    raise NotImplementedError
                self.rel_score.weight = nn.Parameter(weight_1 - weight_2)
            elif isinstance(relevant_init, int):
                weight = self.lm_head.weight.data[relevant_init: relevant_init + 1].detach().clone()
                self.rel_score.weight = nn.Parameter(weight)
            else:
                raise NotImplementedError
        self.post_init()      

model = LlamaForRankCausalLM.from_pretrained("../Llama-2-7b-hf", device_map='cpu')

weight = model.lm_head.weight[5407: 5408].detach().clone()
model.rel_score.weight = nn.Parameter(weight)
import os
os.makedirs("../ckpt/init_ckpt")
model.save_pretrained("../ckpt/init_ckpt/useful", safe_serialization=True)