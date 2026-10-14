from ..sft.trainer import CustomSeq2SeqTrainer
from llmtuner.extras.logging import get_logger

from transformers.utils import is_peft_available, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
import safetensors
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from peft.peft_model import PeftModel
import torch
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from torch import nn
from typing import Dict, Union, Any, Optional, List, Tuple

PREFIX_CHECKPOINT_DIR = "checkpoint"
TRAINER_STATE_NAME = "trainer_state.json"
TRAINING_ARGS_NAME = "training_args.bin"

import os
import numpy as np


logger = get_logger(__name__)


class CustomRankTrainer(CustomSeq2SeqTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir
        if self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                # self.model_wrapped.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
    
    # def _save_checkpoint(self, model, trial, metrics=None):
    #     # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
    #     # want to save except FullyShardedDDP.
    #     # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

    #     # Save model checkpoint
    #     checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    #     if self.hp_search_backend is None and trial is None:
    #         self.store_flos()

    #     run_dir = self._get_output_dir(trial=trial)
    #     output_dir = os.path.join(run_dir, checkpoint_folder)
    #     if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
    #         logger.warning(
    #             f"Checkpoint destination directory {output_dir} already exists and is non-empty."
    #             "Saving will proceed but saved results may be invalid."
    #         )
    #         staging_output_dir = output_dir
    #     else:
    #         staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")
    #     self.save_model(staging_output_dir, _internal_call=True)

    #     if not self.args.save_only_model:
    #         # Save optimizer and scheduler
    #         self._save_optimizer_and_scheduler(staging_output_dir)
    #         # Save RNG state
    #         self._save_rng_state(staging_output_dir)

    #     # Determine the new best metric / best model checkpoint
    #     if metrics is not None and self.args.metric_for_best_model is not None:
    #         metric_to_check = self.args.metric_for_best_model
    #         if not metric_to_check.startswith("eval_"):
    #             metric_to_check = f"eval_{metric_to_check}"
    #         metric_value = metrics[metric_to_check]

    #         operator = np.greater if self.args.greater_is_better else np.less
    #         if (
    #             self.state.best_metric is None
    #             or self.state.best_model_checkpoint is None
    #             or operator(metric_value, self.state.best_metric)
    #         ):
    #             self.state.best_metric = metric_value
    #             self.state.best_model_checkpoint = output_dir

    #     # Save the Trainer state
    #     if self.args.should_save:
    #         self.state.save_to_json(os.path.join(staging_output_dir, TRAINER_STATE_NAME))

    #     if self.args.push_to_hub:
    #         self._push_from_checkpoint(staging_output_dir)

    #     # Place checkpoint in final location after all saving is finished.
    #     # First wait for everyone to finish writing
    #     self.args.distributed_state.wait_for_everyone()
    #     # Then go through the rewriting process starting on process 0
    #     if staging_output_dir != output_dir:
    #         with self.args.main_process_first(
    #             desc="Renaming model checkpoint folder to true location", local=self.args.save_on_each_node
    #         ):
    #             if os.path.exists(staging_output_dir):
    #                 os.rename(staging_output_dir, output_dir)

    #     # Maybe delete some older checkpoints.
    #     if self.args.should_save:
    #         self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:
                inputs["labels"] = inputs["labels"][:, :prompt_len] # truncate the labels instead of padding the inputs

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels