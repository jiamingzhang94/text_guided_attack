#!/bin/bash

# 定义变量
DATA="/home/dycpu6_8tssd1/jmzhang/datasets/"
DATASETS=("snli_ve")
METHODS=("early" "later" "scratch")
LOSSES=("Con" "Cosine")

# 循环执行脚本
for L in "${LOSSES[@]}"; do
    for M in "${METHODS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "---------------------------------------------------"
            echo "Dataset: $DATASET, Image Path: ${DATASET}/${M}_${L} "
            python ve.py \
                --cache_path "$DATA" \
                --cfg_path lavis_tool/albef/snli_eval.yaml \
                --image_path "outputs/${DATASET}/${M}_${L}" \
                --json_path "outputs/${DATASET}_${M}_${L}.json"
        done
    done
done
