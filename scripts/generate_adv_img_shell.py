import os

# 定义变量
method = "early"
loss = "Con"

datasets = [
    "coco_retrieval",
    "flickr30k",
    "snli_ve",
    "coco_caption",
    "nocaps"
]
# 定义数组存储decoder的路径
decoders = [
    f"checkpoints/mel/checkpoints/coco_retrieval_{method}_{loss}.pt",
    f"checkpoints/mel/checkpoints/flickr30k_{method}_{loss}.pt",
    f"checkpoints/mel/checkpoints/flickr30k_{method}_{loss}.pt",
    f"checkpoints/mel/checkpoints/coco_retrieval_{method}_{loss}.pt",
    f"checkpoints/mel/checkpoints/coco_retrieval_{method}_{loss}.pt"
]

# 定义数组存储target_caption的路径
captions = [
    "json/coco_retrieval_target.json",
    "json/flickr30k_target.json",
    "json/snli_ve_target.json",
    "json/coco_caption_target.json",
    "json/nocaps_target.json"
]

paths = [
    "/home/dycpu6_8tssd1/jmzhang/datasets/mscoco",
    "/home/dycpu6_8tssd1/jmzhang/datasets/lavis/flickr30k/images",
    "/home/dycpu6_8tssd1/jmzhang/datasets/lavis/flickr30k/images/flickr30k-images",
    "/home/dycpu6_8tssd1/jmzhang/datasets/mscoco",
    "/home/dycpu6_8tssd1/jmzhang/datasets/lavis/nocaps/images"
]
image_only=[True,True,True,True,True]
is_nocap=[False,False,False,False,True]
task=['retrieval','retrieval','ve','caption','caption']
# image_only=[False,False,False,False,False]
# 循环执行脚本
for i in range(len(datasets)):
    command = (
        f"python generate_adv_img.py "
        f"--model_name 'ViT-B/32' "
        f"--decoder_path '{decoders[i]}' "
        f"--clean_image_path '/home/dycpu6_8tssd1/jmzhang/datasets/ILSVRC2012/val/' "
        f"--target_caption '{captions[i]}' "
        f"--target_image_path '{paths[i]}' "
        f"--batch_size 250 "
        f"--device 'cuda:0' "
        f"--output_path 'outputs' "
        f"--adv_imgs '{method}_{loss}' "
        f"--dataset '{datasets[i]}' "
        f"--task_type {task[i]} "
    )
    if image_only[i]:
        command+=f" --image_only"
    if is_nocap[i]:
        command+=f" --is_nocap"
    command+=f">> logs/generate_{method}_{loss}.log 2>&1"
    os.system(command)