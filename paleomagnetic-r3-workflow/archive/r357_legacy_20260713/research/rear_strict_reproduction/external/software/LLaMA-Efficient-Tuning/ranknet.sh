#!/bin/bash

DEVICE=$1

export CUDA_VISIBLE_DEVICES=${DEVICE}
export OMP_NUM_THREADS=20
llama_path=/home/wangyuhao/self-rag/Llama-2-7b-hf


running_type="26doc-152301"
output_dir=/media/wangyuhao/usere/26doc-152301-u

deepspeed --master_port=9941 src/train_bash.py \
--deepspeed ds_config3_nooff.json \
--stage ranksft \
--model_name_or_path $llama_path \
--do_train \
--save_safetensors true \
--preprocessing_num_worker 10 \
--dataset_dir /mnt/wangyuhao/usere/training \
--dataset $running_type \
--template usere \
--rank_beta 1 \
--psg_num 2 \
--loss_fn bce \
--rank_bias 0.8 \
--relevant_init useful \
--flash_attn true \
--only_rank false \
--finetuning_type full \
--lr_scheduler_type cosine \
--cache_dir /home/wangyuhao/huggingface/datasets/.cache \
--output_dir $output_dir \
--overwrite_output_dir \
--cutoff_len 4090 \
--freeze_lr 0.00001 \
--warmup_ratio 0.03 \
--mask_rel_token true \
--freeze_epoch 0 \
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 2 \
--logging_steps 10 \
--save_steps 400 \
--learning_rate 1e-6 \
--num_train_epochs 2.0 \
--plot_loss \
--bf16 || exit 10086

