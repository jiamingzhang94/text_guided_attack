#!/bin/bash

# 定义变量
method="early"
loss="Con"

# 定义数组存储decoder的路径
decoders=(
    "checkpoints/mel/checkpoints/coco_retrieval_${method}_${loss}.pt"
    "checkpoints/mel/checkpoints/flickr30k_${method}_${loss}.pt"
    "checkpoints/mel/checkpoints/flickr30k_${method}_${loss}.pt"
    "checkpoints/mel/checkpoints/coco_retrieval_${method}_${loss}.pt"
    "checkpoints/mel/checkpoints/coco_retrieval_${method}_${loss}.pt"
)

# 定义数组存储target_caption的路径
captions=(
    "json/coco_retrieval_target.json"
    "json/flickr30k_target.json"
    "json/snli_ve_target.json"
    "json/coco_caption_target.json"
    "json/nocaps_target.json"
)

# 定义数组存储图像路径
paths=(
    "/home/dycpu6_8tssd1/jmzhang/datasets/mscoco"
    "/home/dycpu6_8tssd1/jmzhang/datasets/lavis/flickr30k/images"
    "/home/dycpu6_8tssd1/jmzhang/datasets/lavis/flickr30k/images/flickr30k-images"
    "/home/dycpu6_8tssd1/jmzhang/datasets/mscoco"
    "/home/dycpu6_8tssd1/jmzhang/datasets/lavis/nocaps/images"
)

# 是否只处理图像
image_only=(true true true true true)

# 是否处理nocaps数据集
is_nocap=(false false false false true)

# 任务类型
task=("retrieval" "retrieval" "ve" "caption" "caption")

# 循环执行脚本
for i in "${!decoders[@]}"; do
    command="python generate_adv_img.py \
    --model_name 'ViT-B/32' \
    --decoder_path '${decoders[i]}' \
    --clean_image_path '/home/dycpu6_8tssd1/jmzhang/datasets/ILSVRC2012/val/' \
    --target_caption '${captions[i]}' \
    --target_image_path '${paths[i]}' \
    --batch_size 250 \
    --device 'cuda:0' \
    --output_path 'outputs' \
    --adv_imgs '${method}_${loss}' \
    --dataset '${datasets[i]}' \
    --task_type ${task[i]}"

    if ${image_only[i]}; then
        command+=" --image_only"
    fi

    if ${is_nocap[i]}; then
        command+=" --is_nocap"
    fi

    command+=" >> logs/generate_${method}_${loss}.log 2>&1"

    eval $command
done
