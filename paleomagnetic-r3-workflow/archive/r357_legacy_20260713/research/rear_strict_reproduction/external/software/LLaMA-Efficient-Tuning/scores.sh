#!/bin/bash

DEVICE=$1
RUNNING_TYPE=$2

llama_path=/media/wangyuhao/usere/llama-7b-pureqa-${RUNNING_TYPE}-3000
RUNNING_TYPE=rl-$RUNNING_TYPE

output_dir=/media/wangyuhao/usere/${RUNNING_TYPE}
export_dir=/media/wangyuhao/usere/llama-7b-${RUNNING_TYPE}

export_dir=/media/wangyuhao/usere/llama-7b-test
llama_path=/media/wangyuhao/usere/llama-7b-pureqa-qa-v2-force-3000


export CUDA_VISIBLE_DEVICES=${DEVICE}
export OMP_NUM_THREADS=20


# deepspeed --master_port=9943 src/train_bash.py \
# --deepspeed ds_config.json \
python src/train_bash.py \
--stage rro \
--model_name_or_path $llama_path \
--do_predict \
--per_device_eval_batch_size 8 \
--dataset_dir /mnt/wangyuhao/usere/training \
--dataset ${RUNNING_TYPE} \
--template usere \
--preprocessing_num_worker=10 \
--finetuning_type lora \
--cache_dir /mnt/wangyuhao/usere/huggingface/.cache \
--lora_target q_proj,v_proj \
--output_dir $output_dir \
--overwrite_output_dir \
--resume_lora_training False \
--max_source_length 3090 \
--max_target_length 10 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 1 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 200 \
--learning_rate 1e-5 \
--num_train_epochs 1.0 \
--plot_loss \
--bf16 || exit 1