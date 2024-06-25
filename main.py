import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader
from dataset import MS_COCO,collate_fn
from transformers import CLIPProcessor, CLIPModel

import argparse
from tqdm import tqdm
from model.clip_unet import CLIP_encoder_decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def filter_state_dict(state_dict, filter_keys):
    return {k: v for k, v in state_dict.items() if not any(key in k for key in filter_keys)}

def train(model, dataloader, optimizer,processor,epoch):
    criterion=nn.MSELoss(reduction="sum")
    running_loss=0.0
    for (images,texts) in tqdm(dataloader,desc=f"[{epoch}/{args.epoch}]",position=0):
        optimizer.zero_grad()
        # images=images.to(device)
        inputs=processor(text=[""],images=images,return_tensors="pt",padding=True).to(device)
        outputs,images_encode,outputs_encode=model(inputs)

        # outputs_encode=model.encoder(outputs)
        # outputs_encode=outputs_encode.image_embeds

        loss = criterion(outputs_encode, images_encode)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # break
    average_loss = running_loss / len(dataloader.dataset)
    print(f"Training Loss: {average_loss}")
    return average_loss

#
# def test(model,dataloader,processor):
#     model.eval()
#     with torch.no_grad():
#         for
#     return 0


def main(args):

    model = CLIP_encoder_decoder(args=args)
    processor = CLIPProcessor.from_pretrained(args.clip_model_path)
    # encoder=CLIP_Vision_encoder(args=args)
    model=model.to(device)
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 加载数据集
    train_dataset = MS_COCO(args=args)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle,collate_fn=collate_fn)
    # train(model=model, dataloader=train_data_loader
    for epoch in tqdm(range(args.epoch)):
        train(model=model, dataloader=train_data_loader, optimizer=optimizer,processor=processor,epoch=epoch)
        # 获取模型的 state_dict
        model_state_dict = model.state_dict()

        # 过滤掉包含 "encoder" 的参数
        filtered_state_dict = filter_state_dict(model_state_dict, ['encoder'])

        # 保存过滤后的模型参数
        torch.save(filtered_state_dict, f"save_model/model_epoch_{epoch+1}.pth")
        # torch.save(model.state_dict(), f"save_model/model_epoch_{epoch+1}.pth")
#


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--train", type=str, default=True)
    argparse.add_argument("--clip_model_path", type=str, default="/data2/ModelWarehouse/clip-vit-base-patch32")
    argparse.add_argument("--image_path", type=str, default="/data2/zhiyu/data/coco/images/train2017")
    argparse.add_argument("--data_path", type=str, default="/data2/junhong/proj/text_guide_attack/data/mscoco_exist.parquet")
    argparse.add_argument("--epoch", type=int, default=20)
    argparse.add_argument("--batch_size", type=int, default=256)
    argparse.add_argument("--shuffle", type=bool, default=True)
    args = argparse.parse_args()
    args.device=device
    main(args)
    # print("1")
    # print(0)
    print("ok")