#!/bin/bash

DEVICE=$1
RUNNING_TYPE=$2

# llama_path=/media/public/models/huggingface/meta-llama/Llama-2-7b-hf
RUNNING_TYPE=rl-$RUNNING_TYPE
# checkpoint_dir=/media/wangyuhao/usere/pureqa-qa-v2-force/checkpoint-2000/
# checkpoint_dir=None

output_dir=/media/wangyuhao/usere/${RUNNING_TYPE}
export_dir=/media/wangyuhao/usere/llama-7b-${RUNNING_TYPE}

export_dir=/media/wangyuhao/usere/llama-7b-test
llama_path=/media/wangyuhao/usere/llama-7b-pureqa-qa-v2-force-2000


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
deepspeed --master_port=9945 src/train_bash.py \
--deepspeed ds_config.json \
--stage dpo \
--model_name_or_path $llama_path \
--do_train \
--dataset_dir /mnt/wangyuhao/usere/training \
--dataset ${RUNNING_TYPE} \
--template usere \
--ddp_find_unused_parameters False \
--preprocessing_num_worker=10 \
--finetuning_type lora \
--cache_dir /mnt/wangyuhao/usere/huggingface/.cache \
--lora_target q_proj,v_proj \
--output_dir $output_dir \
--overwrite_output_dir \
--resume_lora_training False \
--dpo_beta 0.5 \
--warmup_steps 600 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 100 \
--learning_rate 1e-6 \
--num_train_epochs 1.0 \
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