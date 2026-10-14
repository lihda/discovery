#!/bin/bash

DEVICE=$1

python change_accelerate_config.py $DEVICE 9900

export CUDA_VISIBLE_DEVICES=${DEVICE}
export OMP_NUM_THREADS=20
llama_path=/home/wangyuhao/self-rag/Llama-2-7b-hf

running_type="26doc-1523dis"
output_dir=/media/wangyuhao/usere/debug

# deepspeed --master_port=9940 src/train_bash.py \
# --deepspeed ds_config3_nooff.json \
# accelerate launch \
#     --config_file config \
# src/train_bash.py \
# deepspeed \
#     --master_port=9940 \
#     --include localhost:${DEVICE} \
    # --deepspeed ds_config3_nooff.json 

python \
src/train_bash.py \
    --stage ranksft \
    --model_name_or_path $llama_path \
    --do_train \
    --rank_smoothing false \
    --save_safetensors true \
    --preprocessing_num_worker 10 \
    --dataset_dir /mnt/wangyuhao/usere/training \
    --dataset $running_type \
    --template usere \
    --psg_num 2 \
    --loss_fn bce \
    --rank_beta 2 \
    --lora_target q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
    --rank_bias 0.8 \
    --relevant_init useful \
    --use_unsloth true \
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
    --freeze_epoch 0 \
    --mask_rel_token true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --save_steps 400 \
    --learning_rate 8e-7 \
    --num_train_epochs 1.5 \
    --plot_loss \
    --bf16 || exit 10086


