#!/bin/bash

CKPT=$1
RUNNING_TYPE=$2
DEVICE=$3

output_dir=/media/wangyuhao/usere/pureqa-${RUNNING_TYPE} #  /checkpoint-${CKPT}
export_dir=/media/wangyuhao/usere/llama-7b-pureqa-select-mul # ${RUNNING_TYPE}-${CKPT}
llama_path=/media/public/models/huggingface/meta-llama/Llama-2-7b-hf
out_file=/mnt/wangyuhao/usere/eval/${RUNNING_TYPE}.json

dirs=("export_dir")
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

python src/export_model.py \
--model_name_or_path $llama_path \
--template usere \
--finetuning_type lora \
--checkpoint_dir $output_dir \
--output_dir $export_dir &&

cp ${llama_path}/generation_config.json ${export_dir} &&

echo "Export model ${output_dir} to ${export_dir}."