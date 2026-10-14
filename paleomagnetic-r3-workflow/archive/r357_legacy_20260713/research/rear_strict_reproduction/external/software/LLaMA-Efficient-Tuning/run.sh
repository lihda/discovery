#!/bin/bash

DEVICE=$1
RUNNING_TYPE=$2

output_dir=/media/wangyuhao/usere/pureqa-${RUNNING_TYPE}-new
export_dir=/media/wangyuhao/usere/llama-7b-pureqa-${RUNNING_TYPE}-new
llama_path=/media/public/models/huggingface/meta-llama/Llama-2-7b-hf
out_file=/mnt/wangyuhao/usere/eval/${RUNNING_TYPE}.json

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
deepspeed --master_port=9942 src/train_bash.py \
--deepspeed ds_config.json \
--stage sft \
--model_name_or_path $llama_path \
--do_train \
--preprocessing_num_worker=10 \
--dataset_dir /mnt/wangyuhao/usere/training \
--dataset ${RUNNING_TYPE} \
--template usere \
--finetuning_type lora \
--cache_dir /mnt/wangyuhao/usere/huggingface/.cache \
--lora_target q_proj,v_proj \
--output_dir $output_dir \
--overwrite_output_dir \
--max_source_length 3090 \
--max_target_length 100 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 500 \
--learning_rate 5e-5 \
--num_train_epochs 2.0 \
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