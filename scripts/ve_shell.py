import os

# 定义变量
DATA = "/home/dycpu6_8tssd1/jmzhang/datasets/"
datasets = ["snli_ve"]
methods = ["early", "later", "scratch"]
losses = ["Con", "Cosine"]

# 循环执行脚本
for l in losses:
    for m in methods:
        for i in range(len(datasets)):
            print("---------------------------------------------------")
            print(f"Dataset: {datasets[i]}, Image Path: {datasets[i]}/{m}_{l} ")
            command = (
                f"python ve.py "
                f"--cache_path {DATA} "
                f"--cfg_path lavis_tool/albef/snli_eval.yaml "#没有需要更换的backbone
                f"--image_path outputs/{datasets[i]}/{m}_{l} "
                f"--json_path outputs/{datasets[i]}_{m}_{l}.json "
            )
            os.system(command)