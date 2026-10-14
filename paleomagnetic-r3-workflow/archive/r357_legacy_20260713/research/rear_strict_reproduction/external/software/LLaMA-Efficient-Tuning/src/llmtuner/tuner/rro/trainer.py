import torch
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union, List
from transformers import BatchEncoding, Trainer
from trl import DPOTrainer
from trl.trainer.utils import disable_dropout_in_model

from llmtuner.extras.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class CustomRROTrainer(DPOTrainer):

    def __init__(
        self,
        beta: float,
        model: Union["PreTrainedModel", torch.nn.Module],
        disable_dropout: Optional[bool] = True,
        loss_type: Optional[Literal["sigmoid", "hinge"]] = "sigmoid",
        neg_per_input: Optional[int] = 1,
        **kwargs
    ):
        if disable_dropout:
            disable_dropout_in_model(model)

        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.use_dpo_data_collator = True # hack to avoid warning
        self.label_pad_token_id = IGNORE_INDEX
        self.loss_type = "sigmoid"
        self.padding_value = 0
        self.beta = beta
        self.neg_per_input = neg_per_input
        self.loss_type = loss_type
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

    def concatenated_forward(
        self,
        model: Optional[torch.nn.Module] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        batch_copied = BatchEncoding({k: v.detach().clone() for k, v in batch.items()}) # avoid error

        all_logits = model(
            input_ids=batch_copied["input_ids"],
            attention_mask=batch_copied["attention_mask"],
            return_dict=True
        ).logits.to(torch.float32)

        all_logps = self._get_batch_logps(
            all_logits,
            batch["labels"],
            average_log_prob=False
        )
        batch_size = batch["input_ids"].size(0) // (1 + self.neg_per_input)
        chosen_logps = all_logps[: batch_size]
        rejected_logps = all_logps[batch_size:]
        chosen_logits = all_logits[: batch_size]
        rejected_logits = all_logits[batch_size:]
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits
    
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        batch_size = batch["input_ids"].size(0) // (1 + self.neg_per_input)
        reference_chosen_logps = batch["scores"][:batch_size]
        reference_rejected_logps = batch["scores"][batch_size:]

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return losses.mean(), metrics