#!/bin/bash

DEVICE=$1
running_type=ranksft-gold

output_dir=/media/wangyuhao/usere/point-search
# export_dir=/media/wangyuhao/usere/llama-7b-$running_type
llama_path=/home/wangyuhao/self-rag/Llama-2-7b-hf

export CUDA_VISIBLE_DEVICES=${DEVICE}
export OMP_NUM_THREADS=20
idx=0
running_type=ranksft-gold
output_dir=/media/wangyuhao/usere/point-search

idx=0
for lr in "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"; do
    idx=$(($idx+1))
    deepspeed --master_port=9944 src/train_bash.py \
    --deepspeed ds_config3_nooff.json \
    --stage ranksft \
    --model_name_or_path $llama_path \
    --do_train \
    --preprocessing_num_worker 10 \
    --dataset_dir /mnt/wangyuhao/usere/training \
    --dataset $running_type \
    --template usere \
    --save_safetensors true \
    --rank_beta 1 \
    --only_rank true \
    --loss_fn bce \
    --rank_bias 1 \
    --flash_attn false \
    --finetuning_type full \
    --lr_scheduler_type constant \
    --cache_dir /home/wangyuhao/huggingface/datasets/.cache \
    --lora_target q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
    --output_dir $output_dir-$idx \
    --overwrite_output_dir \
    --cutoff_len 4090 \
    --freeze_lr 1e-5 \
    --freeze_epoch 0 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --logging_steps 10 \
    --save_steps 400 \
    --max_steps 800 \
    --learning_rate $lr \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16 || exit 10086
done


# cd /home/wangyuhao/usere

# bash split-gen.sh $DEVICE