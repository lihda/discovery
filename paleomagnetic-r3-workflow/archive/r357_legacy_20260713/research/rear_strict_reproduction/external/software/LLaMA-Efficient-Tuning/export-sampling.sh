#!/bin/bash

cd /home/wangyuhao/LLaMA-Efficient-Tuning

CKPT=$1
RUNNING_TYPE=$2
DEVICE=$3
GEN_TYPE=$4
RR=$5
SPLIT=$6

export CUDA_VISIBLE_DEVICES=${DEVICE}
export OMP_NUM_THREADS=20

RUNNING_TYPE=${GEN_TYPE}-${RUNNING_TYPE}
if [ "$CKPT" == "None" ]; then
    export_dir=/media/wangyuhao/usere/llama-7b-pureqa-${RUNNING_TYPE}
    output_dir=/media/wangyuhao/usere/pureqa-${RUNNING_TYPE}
    generate_out=/mnt/wangyuhao/usere/eval/${RUNNING_TYPE}-end-${GEN_TYPE}-${RR}.json
else
    export_dir=/media/wangyuhao/usere/llama-7b-pureqa-${RUNNING_TYPE}-${CKPT}
    output_dir=/media/wangyuhao/usere/pureqa-${RUNNING_TYPE}/checkpoint-${CKPT}
    generate_out=/mnt/wangyuhao/usere/eval/${RUNNING_TYPE}-${CKPT}-${GEN_TYPE}-${RR}.json
fi

llama_path=/media/public/models/huggingface/meta-llama/Llama-2-7b-hf
generate_source=/mnt/wangyuhao/usere/eval/nq-test-pure-${GEN_TYPE}-${RR}-force.json

if [ "$SPLIT" == "train" ]; then
    generate_out=/mnt/wangyuhao/usere/training/${RUNNING_TYPE}-end-${GEN_TYPE}-${RR}.json
    generate_source=/mnt/wangyuhao/usere/training/nq-train-pure-qa-dense-${RUNNING_TYPE}.json
fi

# Check if export_dir exists and contains the safetensors file
if [ -d "$export_dir" ] && [ -f "$export_dir/model-00001-of-00002.safetensors" ]; then
    echo "Model already exported in $export_dir. Skipping export step."
else
    dirs=("export_dir")
    for dir in "${dirs[@]}"; do
        if [ ! -d "${!dir}" ]; then
            mkdir -p "${!dir}"
        else
            rm -rf "${!dir}"
            mkdir -p "${!dir}"
        fi
    done

    python src/export_model.py \
    --model_name_or_path $llama_path \
    --template usere \
    --finetuning_type lora \
    --checkpoint_dir $output_dir \
    --output_dir $export_dir &&

    cp ${llama_path}/generation_config.json ${export_dir} &&

    echo "Successfully export model ${output_dir} to ${export_dir}."
fi &&

cd ../usere &&

python sampling.py \
--model_path $export_dir \
--outfile $generate_out \
--source $generate_source \
--batch_size 12 || exit 1

echo "Generate results to ${generate_out}."

cd ../LLaMA-Efficient-Tuning