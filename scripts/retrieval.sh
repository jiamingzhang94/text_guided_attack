#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
export TORCH_HOME=/home/dycpu6_8tssd1/jmzhang/.cache/
# custom config

#数据集路径
DATA=/home/dycpu6_8tssd1/jmzhang/datasets/

datasets=("coco" "flickr30k")
targets=("clip" "blip" "albef")

for t in "${targets[@]}"; do
    for d in "${datasets[@]}"; do
        python retrieval.py \
        --cache_path ${DATA} \
        --cfg_path ./lavis_tool/${t}/ret_${d}_eval.yaml
    done
done
#        --adv_training \
#                 > logs/retrieval/eval.log 2>&1