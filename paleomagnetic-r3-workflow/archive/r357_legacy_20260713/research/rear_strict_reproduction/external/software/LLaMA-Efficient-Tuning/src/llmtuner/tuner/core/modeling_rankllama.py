""" PyTorch RankLLaMA model."""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import torch.distributions as dist
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import ModelOutput
from peft import PeftModel
import os

from transformers.utils import (
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast, LlamaForCausalLM, LlamaConfig
from dataclasses import dataclass

from torch.nn import BCELoss, BCEWithLogitsLoss
from itertools import product


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, logits, labels):

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


class SftSampler:
    
    def __init__(self, batch_size, group_num, device):
        query_num = batch_size // group_num

        # 设置选取正例的概率
        prob_i_equals_result_i = 0.7

        # 计算每个其他情况的概率
        prob_other = (1 - prob_i_equals_result_i) / (group_num - 1)

        # 创建一个概率分布，满足要求的概率分布
        probabilities = torch.full((query_num, group_num), prob_other, dtype=torch.float, device=device)
        probabilities[:, 0] = prob_i_equals_result_i
        self.prob = probabilities
        self.offset = torch.arange(0, batch_size, group_num, device=device)
        self.m = dist.Categorical(probabilities)

    def __call__(self):
        return self.m.sample() + self.offset
   

class LossForRank:
    
    epsilon = 0.1

    def __init__(
        self,
        loss_fn: str,
        bias: float = 1.0,
        smoothing: bool = False,
        psg_num: int = 4,
        padded_value_indicator: int = -100,
        weight_by_diff: bool = False,
        weight_by_diff_powed: bool = False,
    ):
        self.bias = bias
        self.compute_loss = self.smooth_binary_cross_entropy_with_logits if smoothing else F.binary_cross_entropy_with_logits 
        self.psg_num = psg_num
        self.padded_value_indicator = padded_value_indicator
        self.weight_by_diff = weight_by_diff
        self.weight_by_diff_powed = weight_by_diff_powed
        if loss_fn == 'bce':
            logger.warning("Using loss funtion BCE.")
            self.loss_fn = self.bce
        elif loss_fn == 'pw':
            logger.warning("Using loss funtion Point Wise.")
            self.loss_fn = self.pointwise
        elif loss_fn == 'rn':
            logger.warning("Using loss funtion Rank Net.")
            self.loss_fn = self.ranknet
        elif loss_fn == 'rn+bce':
            logger.warning("Using loss funtion Rank Net with BCE.")
            self.loss_fn = self.rn_with_bce
        elif loss_fn == 'lw':
            logger.warning("Using loss funtion List Wise.")
            self.loss_fn = self.listwise
        else:
            raise NotImplementedError
        
    def __call__(self, **kwargs):
        return self.loss_fn(**kwargs)
    
    def smooth_binary_cross_entropy_with_logits(self, input, target, **kwargs):
        target = target * (1 - self.epsilon) + (1 - target) * self.epsilon
        return F.binary_cross_entropy_with_logits(input, target, **kwargs)
    
    def rn_with_bce(self, **kwargs):
        return (self.ranknet(**kwargs) + self.bce(**kwargs)) / 2
        
    def ranknet(self, scores: torch.FloatTensor, labels: torch.FloatTensor):
        """
        RankNet loss introduced in "Learning to Rank using Gradient Descent".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
        :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
        :return: loss value, a torch.FloatTensor
        """
        y_pred = scores.view(-1, self.psg_num)
        y_true = labels.view(-1, self.psg_num)

        # here we generate every pair of indices from the range of document length in the batch
        document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

        pairs_true = y_true[:, document_pairs_candidates]
        selected_pred = y_pred[:, document_pairs_candidates]

        # here we calculate the relative true relevance of every candidate pair
        true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
        pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

        # here we filter just the pairs that are 'positive' and did not involve a padded instance
        # we can do that since in the candidate pairs we had symetric pairs so we can stick with
        # positive ones for a simpler loss function formulation
        the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

        pred_diffs = pred_diffs[the_mask]

        weight = None
        if self.weight_by_diff:
            abs_diff = torch.abs(true_diffs)
            weight = abs_diff[the_mask]
        elif self.weight_by_diff_powed:
            true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
            abs_diff = torch.abs(true_pow_diffs)
            weight = abs_diff[the_mask]

        # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
        # whether one document is better than the other and not about the actual difference in
        # their relevancy levels
        true_diffs = true_diffs[the_mask]
        true_diffs = (true_diffs > 0).to(pred_diffs.dtype)

        return self.compute_loss(pred_diffs, true_diffs, weight=weight)
    
    def bce(self, scores: torch.FloatTensor, labels: torch.FloatTensor):
        classes = labels[:, 0] >= 0.5 
        pos_scores = scores[classes]
        neg_scores = scores[~classes]
        loss = torch.tensor(0, device=scores.device, dtype=scores.dtype)
        if pos_scores.shape[0]:
            loss += self.compute_loss(pos_scores, torch.ones_like(pos_scores), reduction="sum")
        if neg_scores.shape[0]:
            loss += self.bias * self.compute_loss(neg_scores, torch.zeros_like(neg_scores), reduction="sum")
        loss /= scores.shape[0]
        return loss
    
    def pointwise(self, scores: torch.FloatTensor, labels: torch.FloatTensor):
        return F.binary_cross_entropy_with_logits(scores, labels)
    
    def listwise(self, scores: torch.FloatTensor, **kwargs):
        scores = scores.view(-1, self.psg_num)
        return F.cross_entropy(scores, torch.zeros(scores.shape[0], dtype=torch.long).to(scores.device))

    
@dataclass
class RankCausalLMOutput(ModelOutput):
    rel_scores: torch.FloatTensor = None
    verify_scores: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    past_key_values: torch.Tensor = None
    hidden_states: torch.Tensor = None
    attentions: torch.Tensor = None

class RankMarginRankingLoss:
    def __init__(self, margin: float = 1.0):
        self.margin = margin
    
    def __call__(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        return torch.mean(F.relu(self.margin - pos_scores + neg_scores))


class RankSoftMarginRankingLoss:
    def __init__(self, margin: float = 1.0):
        self.margin = margin
    
    def __call__(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        return torch.mean(F.softplus(self.margin - pos_scores + neg_scores))


class BinaryCrossEntropyLoss:
    def __init__(self, bias: float = 1.0, smoothing: bool = False, **kwargs):
        self.bias = bias
        self.smoothing = smoothing
    def __call__(self, scores: torch.Tensor, classes: torch.tensor):
        pos_scores = scores[classes[:, 0]]
        neg_scores = scores[~classes[:, 0]]
        loss = torch.tensor(0, device=scores.device, dtype=scores.dtype)
        if pos_scores.shape[0]:
            loss += F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores), reduction="sum")
        if neg_scores.shape[0]:
            if self.smoothing:
                neg_labels = torch.full_like(neg_scores, 0.1)
            else:
                neg_labels = torch.zeros_like(neg_scores)
            loss += self.bias * F.binary_cross_entropy_with_logits(neg_scores, neg_labels, reduction="sum")
        loss /= scores.shape[0]
        return loss

class ScorerBinaryCrossEntropyLoss:
    def __call__(self, scores: torch.Tensor, classes: torch.Tensor):
        loss = F.binary_cross_entropy_with_logits(scores, classes)
        return loss
    
class ListWiseLoss:
    def __init__(self, psg_num=4, **kwargs):
        self.psg_num = psg_num
    def __call__(self, scores: torch.Tensor, **kwargs):
        scores = scores.view(-1, self.psg_num)
        return F.cross_entropy(scores, torch.zeros(scores.shape[0], dtype=torch.long).to(scores.device))

class RankCrossEntropyLoss:
    def __call__(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        return (F.cross_entropy(pos_scores, torch.ones(pos_scores.shape[0], dtype=torch.long).to(pos_scores.device)) 
              + F.cross_entropy(neg_scores, torch.zeros(neg_scores.shape[0], dtype=torch.long).to(pos_scores.device)))

# rr_loss_functions = {
#     # "mr": RankMarginRankingLoss,
#     # "smr": RankSoftMarginRankingLoss,
#     "bce": BinaryCrossEntropyLoss,
#     "lw": ListWiseLoss,
#     "rn": RankNetLoss,
#     # "ce": RankCrossEntropyLoss,
# }


class LlamaRankHead(nn.Module):
    def __init__(self, hidden_size, num_labels, bias=False):
        super().__init__()
        self.head_proj = nn.Linear(hidden_size, num_labels, bias=bias)
        self.act_fn = F.softmax

    def forward(self, hidden_state):
        scores = self.head_proj(hidden_state)
        normalised_scores = self.act_fn(scores, dim=1)
        return normalised_scores[:, 0]


class LlamaForRankCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

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
        rank_smoothing: bool = False,
        psg_num: int = 4, 
        rank_only: bool = False,
        ):
        super().__init__(config)

        self.num_labels = num_labels
        self.beta = beta
        
        self.rel_score = nn.Linear(config.hidden_size, num_labels, bias=False)
        self.rel_score_id = rel_score_id

        self.list_wise = False
        self.rank_only = rank_only
        self.rank_loss_fn = LossForRank(
            rr_loss_fn,
            psg_num=psg_num,
            bias=bias,
            smoothing=rank_smoothing)
        self.enable_verify = enable_verify
        self.sft_loss_fn = LabelSmoother() if smoothing else CrossEntropyLoss()
        if self.enable_verify:
            self.score_loss_fn = ScorerBinaryCrossEntropyLoss()
            self.verify_score = nn.Linear(config.hidden_size, num_labels, bias=False)
            self.verify_score_id = verify_score_id
        
        self.post_init()      
  
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        classes: Optional[torch.Tensor] = None,
        ans_score: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        return_score: Optional[bool] = None,
    ) -> Union[Tuple, RankCausalLMOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
            
        if self.beta is None:
            if return_score:
                return self.rank_loss(input_ids=input_ids, outputs=outputs)
            else:
                return self.sft_loss(outputs=outputs, labels=labels)
            
        if self.rank_only:
            return self.rank_loss(input_ids=input_ids, outputs=outputs, classes=classes, ans_score=ans_score)
        
        sft_output = self.sft_loss(outputs=outputs, labels=labels)
        rank_output = self.rank_loss(input_ids=input_ids, outputs=outputs, classes=classes, ans_score=ans_score)
        
        loss = sft_output.loss + self.beta * rank_output.loss
        if torch.isnan(sft_output.loss) or sft_output.loss is None:
            print("sft_output.loss", sft_output.loss)
        if torch.isnan(rank_output.loss) or rank_output.loss is None:
            print("rank_output.loss", rank_output.loss)
        
        if not return_dict:
            output = (sft_output.logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return RankCausalLMOutput(
            loss=loss,
            logits=sft_output.logits,
            rel_scores=rank_output.rel_scores,
            verify_scores=rank_output.verify_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def sft_loss(self, outputs, labels=None, classes=None):
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if labels is None:
            return CausalLMOutputWithPast(
                    loss=torch.tensor(0, device=logits.device),
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
        
        if classes is None:
            # Filter the ones that generate correct labels for sequence to sequence training
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
        else:
            classes = classes.to(torch.bool)
            if sum(classes[:, 1]) == 0:
                return RankCausalLMOutput(loss=torch.tensor(0, device=logits.device))
            shift_logits = logits[classes[:, 1], :-1, :].contiguous()
            shift_labels = labels[classes[:, 1], 1:].contiguous()

        # Flatten the tokens and shifted labels
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss = self.sft_loss_fn(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def rank_loss(self, input_ids=None, outputs=None, classes=None, ans_score=None):

        hidden_states = outputs[0]
        batch_size = hidden_states.size()[0]

        rel_position = (torch.eq(input_ids, self.rel_score_id).long().argmax(-1)).to(
            hidden_states.device
        )
        rel_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), rel_position]
        rel_logits = self.rel_score(rel_hidden_states)
        
        loss = None
        if classes is not None:
            non_zero_positions = rel_position != 0
            filtered_rel_logits = rel_logits[non_zero_positions]
            filtered_rel_classes = classes[non_zero_positions]
            loss = self.rank_loss_fn(scores=filtered_rel_logits, labels=filtered_rel_classes[:, :1].to(filtered_rel_logits.dtype))
        
        return RankCausalLMOutput(
            loss=loss,
            rel_scores=rel_logits,
            verify_scores=rel_logits,
        )
    
            # if self.enable_verify:
            #     verify_position = (torch.eq(input_ids, self.verify_score_id).long().argmax(-1)).to(
            #         hidden_states.device
            #     )
            #     verify_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), verify_position]
            #     non_zero_positions = verify_position != 0
            #     verify_logits = self.verify_score(verify_hidden_states)
            #     verify_logits = verify_logits[non_zero_positions]
            #     verify_classes = classes[non_zero_positions]
            # else:
            #     verify_logits = torch.ones_like(rel_logits, device=rel_logits.device)

            # if self.enable_verify:
            #     if ans_score is None:
            #         pos_verify_logits = verify_logits[classes[:, 1]]
            #         neg_verify_logits = verify_logits[~classes[:, 1]]
            #         verify_loss = self.rank_loss_fn(pos_verify_logits, neg_verify_logits)
            #     else:
            #         verify_loss = self.score_loss_fn(verify_logits, ans_score)
            #     loss += verify_loss

        # return RankCausalLMOutput(
        #     loss=loss,
        #     rel_scores=rel_logits,
        #     verify_scores=verify_logits,
        # )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        rel_score_weight_init = kwargs.pop("rel_score_weight_init", None)
        if rel_score_weight_init is not None and os.path.exists(f"/mnt/wangyuhao/usere/{rel_score_weight_init}"):
            pretrained_model_name_or_path = f"../ckpt/init_ckpt/{rel_score_weight_init}"
            print(f"Init model with: {pretrained_model_name_or_path}")
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        # if rel_score_weight_init is not None:
        #     import os,pdb
        #     if os.environ["LOCAL_RANK"] == 0:
        #     pdb.set_trace()
        #     model = PeftModel.from_pretrained(model, f"/mnt/wangyuhao/usere/{rel_score_weight_init}")
        #     model.merge_and_unload()
        return model
    

