import os

# 定义变量
DATA = "/home/dycpu6_8tssd1/jmzhang/datasets/"
backbones = ["vitb32", "vitb16", "vitl14"]

datasets = ["coco_retrieval", "flickr30k"]

methods = ["early", "later", "scratch"]
losses = ["Con", "Cosine"]

# 循环执行脚本
for l in losses:
    for m in methods:
        for b in backbones:
            for i in range(len(datasets)):
                print("---------------------------------------------------")
                print(f"Target model: {b}")
                print(f"Dataset: {datasets[i]}, Image Path: {datasets[i]}/{m}_{l} ")

                command = (
                    f"python retrieval.py "
                    f"--cache_path {DATA} "
                    f"--cfg_path lavis_tool/clip/ret_coco_retrieval_eval_{b}.yaml "
                    f"--image_path outputs/{datasets[i]}/{m}_{l} "
                    f"--json_path outputs/{datasets[i]}_{m}_{l}.json "
                )

                os.system(command)
