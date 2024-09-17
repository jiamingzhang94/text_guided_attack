import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4,0"
from diffusers import DDPMPipeline
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline
import torch
from tqdm import trange
# from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("/new_data/yifei2/junhong/AttackVLM-main/model/Fluently-XL-v2",device_map="balanced")

# # Step 3: Move the pipeline to GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# pipeline.to(device)

# Step 4: Generate an image
# Since you did not specify what kind of input your model requires,
# I will assume it's a text-to-image model and generate an image based on a prompt.
generator = torch.Generator().manual_seed(42)  # For reproducibility
with open("/new_data/yifei2/junhong/AttackVLM-main/data/captions/coco_captions_1000.txt",'r',encoding="utf-8") as f:
    captions=f.readlines()
# 使用trange来添加进度条
for idx in trange(len(captions), desc='Generating Images'):
    caption = captions[idx].strip()
    prompt = caption
    image = pipeline(prompt=prompt, generator=generator).images[0]
    image.save(f"/new_data/yifei2/junhong/AttackVLM-main/data/fluently_generate/samples/{idx:05d}.png")
