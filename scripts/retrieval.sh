#!/bin/bash

export CUDA_VISIBLE_DEVICES=4
export TORCH_HOME=/home/dycpu6_8tssd1/jmzhang/.cache/
#export TORCH_HOME=/new_data/yifei2/junhong/AttackVLM-main/model/blip-cache
# custom config

#数据集路径
DATA=/home/dycpu6_8tssd1/jmzhang/datasets/
#DATA=/new_data/yifei2/junhong/dataset

datasets=("coco" "flickr")
targets=("clip" "blip" "albef")
image_path=("/YOUR/COCO/images" "/YOUR/flickr/images")
#image_paths=("/new_data/yifei2/junhong/dataset/new_coco/coco/images" "/new_data/yifei2/junhong/dataset/flickr30k/flickr30k-images")

# 使用索引变量 i 遍历两个数组
for t in "${targets[@]}"; do
    for i in "${!datasets[@]}"; do
        d=${datasets[$i]}
        image_path=${image_paths[$i]}
        echo "Dataset: ${d}, Image Path: ${image_path}"
        python ../retrieval.py \
            --cache_path ${DATA} \
            --cfg_path ../lavis_tool/${t}/ret_${d}_eval.yaml \
            --image_path ${image_path}
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