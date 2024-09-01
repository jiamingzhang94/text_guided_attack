#!/bin/bash

# 定义变量
DATA="/home/dycpu6_8tssd1/jmzhang/datasets/"
backbones=("vitb32" "vitb16" "vitl14")
datasets=("coco_retrieval" "flickr30k")
methods=("early" "later" "scratch")
losses=("Con" "Cosine")

# 循环执行脚本
for l in "${losses[@]}"; do
    for m in "${methods[@]}"; do
        for b in "${backbones[@]}"; do
            for i in "${!datasets[@]}"; do
                echo "---------------------------------------------------"
                echo "Target model: ${b}"
                echo "Dataset: ${datasets[$i]}, Image Path: outputs/${datasets[$i]}/${m}_${l}"

                python retrieval.py \
                    --cache_path ${DATA} \
                    --cfg_path "lavis_tool/clip/ret_coco_retrieval_eval_${b}.yaml" \
                    --image_path "outputs/${datasets[$i]}/${m}_${l}" \
                    --json_path "outputs/${datasets[$i]}_${m}_${l}.json"
            done
        done
    done
done
