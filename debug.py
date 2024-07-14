import webdataset as wds
from torch.utils.data import DataLoader
import glob
import os

import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
import models.clip as clip
import torch
from PIL import Image


# tar_dir = "/home/dycpu6_8tssd1/jmzhang/datasets/coco/mscoco"
# tar_files = glob.glob(os.path.join(tar_dir, "*.tar"))
#
#
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
#
# def handle_sample(sample):
#     try:
#         image, text = sample
#         return train_transform(image), text
#     except Exception as e:
#         logging.error(f"Error processing sample: {e}")
#         return None
#
#
# dataset = (
#     wds.WebDataset(tar_files)
#     .decode("pil")
#     .to_tuple("jpg", "txt")
#     .map(handle_sample)
#     .batched(32)
# )
#
# # 创建 DataLoader
# dataloader = DataLoader(dataset, batch_size=None, num_workers=4)
#
# # 遍历数据
# for images, data in dataloader:
#     print(images.shape, data)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("/homes/jmzhang/000000000.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    print(1)