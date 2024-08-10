#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
export TORCH_HOME=/home/dycpu6_8tssd1/jmzhang/.cache/
# custom config

#数据集路径
DATA=/home/dycpu6_8tssd1/jmzhang/datasets/

#datasets=("coco" "flickr30k")
datasets=("coco")
targets=("clip" "blip" "albef")
#image_path=("/YOUR/COCO/PATH","YOUR/FLICKER/PATH")
image_path=("/home/dycpu6_8tssd1/jmzhang/datasets/mscoco/")

# 使用索引变量i来遍历两个数组
for t in "${targets[@]}"; do
    for ((i=0; i<${#datasets[@]}; i++)); do
        d="${datasets[$i]}"
        image="${image_path[$i]}"
        python retrieval.py \
        --cache_path ${DATA} \
        --cfg_path ./lavis_tool/${t}/ret_${d}_eval.yaml \
        --image_path "${image}"
    done
done

#for t in "${targets[@]}"; do
#    for d in "${datasets[@]}"; do
#        python retrieval.py \
#        --cache_path ${DATA} \
#        --cfg_path ./lavis_tool/${t}/ret_${d}_eval.yaml \
#        --image_path ${IMAGE_PATH}
#    done
#done
#        --adv_training \
#                 > logs/retrieval/eval.log 2>&1