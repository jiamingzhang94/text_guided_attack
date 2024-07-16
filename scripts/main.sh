#!/bin/bash

# Set environment variables if necessary
export MASTER_PORT=23456
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Command to run the training script with torchrun
torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} main_ddp.py \
    --tar_dir="/home/dycpu6_8tssd1/jmzhang/datasets/coco/mscoco" \
    --epoch=40 \
    --batch_size=350 \
    --dist_url="tcp://127.0.0.1:${MASTER_PORT}" \
    --checkpoint='/home/dycpu6_8tssd1/jmzhang/codes/text_guided_attack/checkpoints/model_epoch_19.pt' \
    > logs/mscoco_1e4_20-40.log 2>&1

# --tar_dir="/home/dycpu4_data1/jmzhang/big_datasets/laion-400m/laion400m-data" \