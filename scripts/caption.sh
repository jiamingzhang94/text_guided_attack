#!/bin/bash

# 定义变量
DATA="/home/dycpu6_8tssd1/jmzhang/datasets/"
DATASETS=("coco_caption" "nocaps")
METHODS=("early" "later" "scratch")
LOSSES=("Con" "Cosine")

# 循环执行脚本
for L in "${LOSSES[@]}"; do
    for M in "${METHODS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "---------------------------------------------------"
            echo "Dataset: $DATASET, Image Path: ${DATASET}/${M}_${L} "
            COMMAND="python caption.py --cache_path $DATA --image_path outputs/${DATASET}/${M}_${L} --json_path outputs/${DATASET}_${M}_${L}.json --gt_path /YOUR/GROUND/TRUTH/PATH"
            if [ "$DATASET" = "coco_caption" ]; then
                COMMAND+=" --cfg_path lavis_tool/blip/caption_coco_eval.yaml"
            elif [ "$DATASET" = "nocaps" ]; then
                COMMAND+=" --cfg_path lavis_tool/blip/caption_nocaps_eval.yaml"
            fi
            eval $COMMAND
        done
    done
done
