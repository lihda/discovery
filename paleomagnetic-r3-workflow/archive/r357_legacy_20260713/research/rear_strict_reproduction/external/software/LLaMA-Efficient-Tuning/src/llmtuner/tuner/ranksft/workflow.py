# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/summarization/run_summarization.py

from typing import TYPE_CHECKING, Optional, List
from transformers import Seq2SeqTrainingArguments
import torch
from .collator import RanksftDataCollator

from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import get_logits_processor
from llmtuner.extras.ploting import plot_loss
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.sft.metric import ComputeMetrics
from .trainer import CustomRankTrainer
from deepspeed.accelerator import get_accelerator

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments


def run_ranksft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):  
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="ranksft")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="ranksft" if (finetuning_args.loss_fn=="bce" or finetuning_args.loss_fn=="pw") else "listsft")
        
    device = model.device

    is_rank = (model.beta is not None) and (finetuning_args.freeze_epoch > 0) and (training_args.resume_from_checkpoint is None)

    if is_rank:
        for param in model.model.parameters(): 
            param.requires_grad_(False)
        for param in model.lm_head.parameters(): 
            param.requires_grad_(False)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left" # use left-padding in generation
    else:
        tokenizer.padding_side = "right"

    data_collator = RanksftDataCollator(
        tokenizer=tokenizer,
        mask_rel_token=finetuning_args.mask_rel_token,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None, # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.cutoff_len,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    if finetuning_args.loss_fn != "bce" and finetuning_args.loss_fn != "pw":
        training_args_dict["per_device_train_batch_size"] //= finetuning_args.psg_num
    training_args_dict.update(dict(remove_unused_columns=False)) # important for pairwise dataset
    training_args = Seq2SeqTrainingArguments(**training_args_dict)
    training_args_dict.pop("warmup_ratio")
    training_args_dict.pop("lr_scheduler_type")
    training_args_dict.update(dict(learning_rate=finetuning_args.freeze_lr, num_train_epochs=finetuning_args.freeze_epoch, save_steps=3000, output_dir=training_args_dict["output_dir"] + "-freeze")) 
    training_head_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Initialize our Trainer
    trainer = CustomRankTrainer(
        model=model,
        args=training_head_args if is_rank else training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **split_dataset(dataset, data_args, training_args)
    )

    # Training
    if training_args.do_train:
        if is_rank:
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            model = trainer.model.to(device)
            for param in model.model.parameters(): 
                param.requires_grad_(True)
            for param in model.lm_head.parameters(): 
                param.requires_grad_(True)
            get_accelerator().empty_cache()

            trainer.save_state()
            trainer.save_model()
            del trainer

            trainer = CustomRankTrainer(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=callbacks,
                **split_dataset(dataset, data_args, training_args)
            )

        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    if training_args.do_eval or training_args.do_predict:
        # Keyword arguments for `model.generate`
        gen_kwargs = generating_args.to_dict()
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        gen_kwargs["logits_processor"] = get_logits_processor()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate: # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate: # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

