import os

# 定义变量
DATA = "/home/dycpu6_8tssd1/jmzhang/datasets/"
datasets = [ "coco_caption","nocaps"]
methods = ["early", "later", "scratch"]
losses = ["Con", "Cosine"]

# 循环执行脚本
for l in losses:
    for m in methods:
        for i in range(len(datasets)):
            print("---------------------------------------------------")
            print(f"Dataset: {datasets[i]}, Image Path: {datasets[i]}/{m}_{l} ")

            command = (
                f"python caption.py "
                f"--cache_path {DATA} "
                f"--image_path outputs/{datasets[i]}/{m}_{l} "
                f"--json_path outputs/{datasets[i]}_{m}_{l}.json "
                f"--gt_path /YOUR/GROUND/TRUTH/PATH" # ground_truth.json的路径
            )
            if datasets[i] == "coco_caption":
                command+=f" --cfg_path lavis_tool/blip/caption_coco_eval.yaml "#没有需要更换的backbone
            elif datasets[i]== "nocaps":
                command+=f" --cfg_path lavis_tool/blip/caption_nocaps_eval.yaml "
            os.system(command)