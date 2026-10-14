#!/bin/bash

DEVICE=$1
running_type=ranksft-list

output_dir=/media/wangyuhao/usere/list-search
# export_dir=/media/wangyuhao/usere/llama-7b-$running_type-2
llama_path=/home/wangyuhao/self-rag/Llama-2-7b-hf
# llama_path=/media/wangyuhao/usere/ranksft-hard-2-2-freeze

export CUDA_VISIBLE_DEVICES=${DEVICE}
export OMP_NUM_THREADS=20
idx=-1
for lr in "5e-4" "1e-4" "5e-5" "1e-5" "5e-6" "1e-6"; do
    idx=$(($idx+1))
    deepspeed --master_port=9943 src/train_bash.py \
    --deepspeed ds_config3_nooff.json \
    --stage ranksft \
    --model_name_or_path $llama_path \
    --do_train \
    --loss_fn lw \
    --preprocessing_num_worker=10 \
    --dataset_dir /mnt/wangyuhao/usere/training \
    --dataset $running_type \
    --template usere \
    --flash_attn false \
    --finetuning_type full \
    --cache_dir /home/wangyuhao/huggingface/datasets/.cache \
    --output_dir $output_dir-$idx \
    --overwrite_output_dir \
    --cutoff_len 4090 \
    --lr_scheduler_type constant \
    --max_steps 800 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 10 \
    --save_steps 400 \
    --list_wise true \
    --rank_beta 50 \
    --psg_num 4 \
    --learning_rate $lr \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16 || exit 10086
done

# echo 'The training and inference processes are all done perfectly.'


# cd /home/wangyuhao/usere

# bash split-gen.sh $DEVICE