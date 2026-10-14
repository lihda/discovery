#!/bin/bash

DEVICE=$1
# RUNNING_TYPE=$2

output_dir=/media/wangyuhao/usere/self-rag-v1-DEVICE
export_dir=/media/wangyuhao/usere/llama-7b-ranksft-DEVICE
llama_path=/home/wangyuhao/self-rag/Llama-2-7b-hf
out_file=/mnt/wangyuhao/usere/eval/self-rag-v1.json

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
deepspeed --master_port=9940 src/train_bash.py \
--deepspeed ds_config2.json \
--stage ranksft \
--model_name_or_path $llama_path \
--do_train \
--preprocessing_num_worker=10 \
--dataset_dir /mnt/wangyuhao/usere/training \
--dataset ranksft \
--template usere \
--flash_attn false \
--finetuning_type lora \
--cache_dir /home/wangyuhao/huggingface/datasets/.cache \
--lora_target q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
--output_dir $output_dir \
--overwrite_output_dir \
--cutoff_len 4090 \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 2 \
--lr_scheduler_type linear \
--logging_steps 10 \
--save_steps 100 \
--warmup_ratio 0.03 \
--learning_rate 2e-5 \
--num_train_epochs 3.0 \
--plot_loss \
--bf16 &&

python src/export_model.py \
--model_name_or_path $llama_path \
--template usere \
--finetuning_type lora \
--checkpoint_dir $output_dir \
--output_dir $export_dir &&

cp ${llama_path}/generation_config.json ${export_dir} &&

python /home/wangyuhao/usere/get_res.py \
--source=/mnt/wangyuhao/usere/corpus/nq/nq-test-col.json  \
--outfile=$out_file \
--model_path=$export_dir \
--type=qa --batch_size=4 || exit 1

echo 'The training and inference processes are all done perfectly.'