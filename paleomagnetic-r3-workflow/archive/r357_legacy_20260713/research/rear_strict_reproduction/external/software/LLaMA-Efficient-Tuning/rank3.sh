#!/bin/bash

DEVICE=$1

export OMP_NUM_THREADS=20
llama_path=../Llama-2-7b-hf

running_type=26doc-1523-grouped
output_dir=26doc-1523-grouped-rbu

deepspeed \
    --master_port=9943 \
    src/train_bash.py \
    --deepspeed ds_z3.json \
    --stage ranksft \
    --model_name_or_path $llama_path \
    --do_train \
    --rank_smoothing false \
    --save_safetensors true \
    --preprocessing_num_worker 10 \
    --dataset_dir ../data/training \
    --dataset $running_type \
    --template usere \
    --psg_num 8 \
    --loss_fn rn+bce \
    --rank_beta 1 \
    --rank_bias 0.8 \
    --relevant_init useful \
    --use_unsloth false \
    --flash_attn false \
    --only_rank false \
    --finetuning_type full \
    --lr_scheduler_type cosine \
    --output_dir ../ckpt/$output_dir \
    --overwrite_output_dir \
    --cutoff_len 4090 \
    --freeze_lr 0.00001 \
    --warmup_ratio 0.03 \
    --freeze_epoch 0 \
    --mask_rel_token true \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 1e-6 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --bf16 || exit 10086

running_type=26doc-1523-grouped
output_dir=26doc-1523-grouped-rnu

deepspeed \
    --master_port=9943 \
    src/train_bash.py \
    --deepspeed ds_z3.json \
    --stage ranksft \
    --model_name_or_path $llama_path \
    --do_train \
    --rank_smoothing false \
    --save_safetensors true \
    --preprocessing_num_worker 10 \
    --dataset_dir ../data/training \
    --dataset $running_type \
    --template usere \
    --psg_num 8 \
    --loss_fn rn \
    --rank_beta 1 \
    --rank_bias 0.8 \
    --relevant_init useful \
    --use_unsloth false \
    --flash_attn false \
    --only_rank false \
    --finetuning_type full \
    --lr_scheduler_type cosine \
    --output_dir ../ckpt/$output_dir \
    --overwrite_output_dir \
    --cutoff_len 4090 \
    --freeze_lr 0.00001 \
    --warmup_ratio 0.03 \
    --freeze_epoch 0 \
    --mask_rel_token true \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 1e-6 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --bf16 || exit 10086

