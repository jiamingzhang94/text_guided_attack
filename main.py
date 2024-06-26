import json
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader
from dataset import MS_COCO, collate_fn
from transformers import CLIPProcessor, CLIPModel

import argparse
from tqdm import tqdm
from model.clip_unet import CLIP_encoder_decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def filter_state_dict(state_dict, filter_keys):
    return {k: v for k, v in state_dict.items() if not any(key in k for key in filter_keys)}
def load_model(model, model_path):
    state_dict = torch.load(model_path)
    filtered_state_dict = filter_state_dict(state_dict, ['encoder'])
    model.load_state_dict(filtered_state_dict, strict=False)
    return model

def train(model, dataloader, optimizer, processor, epoch):
    criterion = nn.MSELoss(reduction="sum")
    running_loss = 0.0
    for (images, texts) in tqdm(dataloader, desc=f"[{epoch}/{args.epoch}]", position=0):
        optimizer.zero_grad()
        # images=images.to(device)
        inputs = processor(text=[""], images=images, return_tensors="pt", padding=True).to(device)
        outputs, images_encode, outputs_encode = model(inputs)

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
def test(model, dataloader, processor):
    def process_text(texts):
        input_text = []
        for text in texts:
            input_text.append(text[0])
        return input_text

    def calculate_cos(image_emb, output_encode):
        cos_sim = torch.nn.functional.cosine_similarity(image_emb, output_encode, dim=1)
        return cos_sim.sum()

    sim_origin=0.0
    sim_images=0.0
    sim_texts=0.0
    sim_image_text=0.0
    sim_text_image_outputs_encode=0.0
    count_batch=0
    eval_batch=200
    model.eval()  # 确保模型在评估模式
    with torch.no_grad():  # 关闭梯度计算
        for (images, texts) in tqdm(dataloader, desc="Testing", position=0):
            inputs = processor(text=process_text(texts), images=images, return_tensors="pt", padding=True).to(device)
            # outputs ：输出的噪声
            # text_encode:输入的文本编码
            # text_output_encode:噪声的编码
            image_outputs, images_encode, image_outputs_encode = model(inputs)
            text_outputs, text_encode, text_outputs_encode = model(inputs, train=False)

            sim_origin+=calculate_cos(images_encode, text_encode)
            sim_images+=calculate_cos(images_encode, image_outputs_encode)
            sim_texts+=calculate_cos(text_encode, text_outputs_encode)
            sim_image_text+=calculate_cos(images_encode,text_outputs_encode)
            sim_text_image_outputs_encode+=calculate_cos(text_outputs_encode, image_outputs_encode)
            count_batch+=1
            if count_batch==eval_batch:
                break
    print("avg_sim_origin:",sim_origin/(eval_batch*args.batch_size))
    print("avg_sim_images:",sim_images/(eval_batch*args.batch_size))
    print("avg_sim_texts:",sim_texts/(eval_batch*args.batch_size))
    print("avg_sim_image_text:",sim_image_text/(eval_batch*args.batch_size))
    print("avg_sim_text_image_outputs_encode:",sim_text_image_outputs_encode/(eval_batch*args.batch_size))
    data=[
        {
            "avg_sim_origin:":sim_origin.item() / (eval_batch * args.batch_size),
            "avg_sim_images:": sim_images.item() / (eval_batch * args.batch_size),
            "avg_sim_texts:": sim_texts.item() / (eval_batch * args.batch_size),
            "avg_sim_image_text:": sim_image_text.item() / (eval_batch * args.batch_size),
            "avg_sim_text_image_outputs_encode:": sim_text_image_outputs_encode.item() / (eval_batch * args.batch_size),
        }
    ]
    save_name=args.model_path.split("/")[-1].split(".")[0]
    with open(f"save_result/{save_name}.json","w",encoding="utf-8")as f:
        json.dump(data,f,ensure_ascii=False,indent=4)
    # return 0


def main(args):
    model = CLIP_encoder_decoder(args=args)
    processor = CLIPProcessor.from_pretrained(args.clip_model_path)
    # encoder=CLIP_Vision_encoder(args=args)
    model = model.to(device)
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 加载数据集
    train_dataset = MS_COCO(args=args)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                                   collate_fn=collate_fn)
    # train(model=model, dataloader=train_data_loader
    if args.mode == "train":
        for epoch in tqdm(range(args.epoch)):
            train(model=model, dataloader=train_data_loader, optimizer=optimizer, processor=processor, epoch=epoch)
            # 获取模型的 state_dict
            model_state_dict = model.state_dict()

            # 过滤掉包含 "encoder" 的参数
            filtered_state_dict = filter_state_dict(model_state_dict, ['encoder'])

            # 保存过滤后的模型参数
            torch.save(filtered_state_dict, f"save_model/model_epoch_{epoch + 1}.pth")
            # torch.save(model.state_dict(), f"save_model/model_epoch_{epoch+1}.pth")
    elif args.mode == "test":
        # for epoch in tqdm(range(args.epoch)):
        # model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("="*100)
        print(args.model_path)
        model=load_model(model,args.model_path)
        test(model=model, dataloader=train_data_loader, processor=processor)

    else:
        print("train or test")


#


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--train", type=str, default=True)
    argparse.add_argument("--clip_model_path", type=str, default="/data2/ModelWarehouse/clip-vit-base-patch32")
    argparse.add_argument("--image_path", type=str, default="/data2/zhiyu/data/coco/images/train2017")
    argparse.add_argument("--data_path", type=str,
                          default="/data2/junhong/proj/text_guide_attack/data/mscoco_exist.parquet")
    argparse.add_argument("--epoch", type=int, default=20)
    argparse.add_argument("--batch_size", type=int, default=256)
    argparse.add_argument("--shuffle", type=bool, default=True)
    argparse.add_argument("--mode", type=str, default="test")
    argparse.add_argument("--model_path", type=str, default="save_model/model_epoch_20.pth")
    argparse.add_argument("--epsilon",type=float,default=8)
    args = argparse.parse_args()
    args.device = device
    main(args)
    # print("1")
    # print(0)
    # print("ok")
