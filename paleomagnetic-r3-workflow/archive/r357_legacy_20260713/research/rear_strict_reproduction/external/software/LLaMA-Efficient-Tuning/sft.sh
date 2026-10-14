#!/bin/bash

DEVICE=$1
running_type=ranksft-hard-2

output_dir=/media/wangyuhao/usere/sft
export_dir=/media/wangyuhao/usere/llama-7b-sft
llama_path=/home/wangyuhao/self-rag/Llama-2-7b-hf

dirs=("output_dir" "export_dir")
for dir in "${dirs[@]}";do
    if [ ! -d "${!dir}" ]; then
        mkdir -p "${!dir}"
    else
        rm -rf "${!dir}"
        mkdir -p "${!dir}"
    fi
done
export CUDA_VISIBLE_DEVICES=${DEVICE}
export OMP_NUM_THREADS=20
deepspeed --master_port=9941 src/train_bash.py \
--deepspeed ds_config.json \
--stage ranksft \
--model_name_or_path $llama_path \
--do_train \
--preprocessing_num_worker 10 \
--dataset_dir /mnt/wangyuhao/usere/training \
--dataset $running_type \
--template usere \
--flash_attn false \
--finetuning_type full \
--cache_dir /home/wangyuhao/huggingface/datasets/.cache \
--lora_target q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
--output_dir $output_dir \
--overwrite_output_dir \
--cutoff_len 4090 \
--per_device_train_batch_size 128 \
--gradient_accumulation_steps 2 \
--lr_scheduler_type linear \
--logging_steps 10 \
--save_steps 100 \
--warmup_ratio 0.03 \
--learning_rate 2e-5 \
--num_train_epochs 3.0 \
--plot_loss \
--bf16

echo 'The training and inference processes are all done perfectly.'


# cd /home/wangyuhao/usere

# bash split-gen.sh $DEVICE