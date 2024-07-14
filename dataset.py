import torch
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import requests
from io import BytesIO
import argparse
import os
import tarfile
import io
from torchvision import transforms
import matplotlib.pyplot as plt
# class MS_COCO(Dataset):
#     def __init__(self, args):
#         self.args = args
#         self.data_list=pq.read_table(args.data_path)
#         self.data_list=self.data_list.to_pandas()
#         self.image_path=args.image_path
#
#     def __len__(self):
#         return len(self.data_list)
#
#     def __getitem__(self, idx):
#
#         text = self.data_list['TEXT'][idx]
#
#         image_name = self.data_list['URL'][idx]
#
#         img_path=os.path.join(self.image_path,image_name)
#         img = Image.open(img_path).convert('RGB')
#
#         # You can perform additional transformations on the image here if needed
#
#         return img, text
#
# def collate_fn(batch):
#     images, texts = zip(*batch)
#     images = list(images)
#     # texts = list(texts)
#     texts=[text.tolist() for text in texts]
#     return images, texts
class image_description_dataset(Dataset):
    def __init__(self, tar_paths,transform=None):
        self.tar_paths = tar_paths
        self.transform = transform
        self.image_files = []
        self.text_files = {}

        for tar_path in self.tar_paths:
            with tarfile.open(tar_path, 'r') as tar:
                img_files = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.jpg')]
                txt_files = {m.name: m for m in tar.getmembers() if m.isfile() and m.name.endswith('.txt')}
                self.image_files.extend([(tar_path, img_file) for img_file in img_files])
                self.text_files.update({(tar_path, m.name): m for m in txt_files.values()})

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        tar_path, img_file = self.image_files[idx]
        with tarfile.open(tar_path, 'r') as tar:
            img_bytes = tar.extractfile(img_file).read()
            img = Image.open(io.BytesIO(img_bytes))
            if self.transform:
                img = self.transform(img)

            text_file_name = img_file.name.replace('.jpg', '.txt')
            if (tar_path, text_file_name) in self.text_files:
                text_file = self.text_files[(tar_path, text_file_name)]
                text_bytes = tar.extractfile(text_file).read()
                description = text_bytes.decode('utf-8').strip()
            else:
                description = ""

        return img, description

def collate_fn(batch):
    images, texts = zip(*batch)
    images = list(images)
    texts = list(texts)
    # texts=[text.tolist() for text in texts]
    return images, texts
# 筛选出tar文件
def get_tar_files(folder_path):
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.tar')]

if __name__== '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--train", type=str, default=True)
    argparse.add_argument("--clip_model_path", type=str, default="/data2/ModelWarehouse/clip-vit-base-patch32")
    argparse.add_argument("--image_path", type=str, default="/data2/zhiyu/data/coco/images/train2017")
    argparse.add_argument("--data_path", type=str, default="/data2/junhong/proj/text_guide_attack/data/mscoco_exist.parquet")
    args = argparse.parse_args()
    # ms_coco=MS_COCO(args)
    # print(len(ms_coco))
    # # Example DataLoader parameters
    # batch_size = 32
    # shuffle = True
    # # Create DataLoader
    # data_loader = DataLoader(ms_coco, batch_size=batch_size, shuffle=shuffle,collate_fn=collate_fn)
    # # Example usage in training loop
    # # print(len(ms_coco))
    # # train2017=os.listdir("/data2/zhiyu/data/coco/images/train2017")
    # # print(len(train2017))
    # num_batches_to_iterate=5
    # for batch_idx, (images, texts) in enumerate(data_loader):
    #     if batch_idx >= num_batches_to_iterate:
    #         break
    #     print(f"Batch {batch_idx}, Number of samples: {len(images)}")
    #     print(f"Text: {texts[0]}")  # Print text description
    # # print(ms_coco.__getitem__(98397))
    tar_files = get_tar_files("F:/dataset/sbu-captions-all/data")
    tar_files = tar_files[:1]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # 创建Dataset和DataLoader
    dataset = image_description_dataset(tar_paths=tar_files, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


    # 使用DataLoader进行迭代和可视化
    # 使用DataLoader进行迭代和可视化
    def show_images(images, descriptions):
        batch_size = len(images)
        rows = (batch_size + 1) // 2  # 动态计算行数
        plt.figure(figsize=(10, 10))
        for i in range(batch_size):
            img = images[i].permute(1, 2, 0).numpy()  # 转换为 HWC 格式并转为 numpy 数组
            plt.subplot(rows, 2, i + 1)
            plt.imshow(img)
            # plt.title(descriptions[i])
            plt.axis('off')
        plt.show()


    # 仅展示一个批次的图像和描述
    for images, descriptions in dataloader:
        show_images(images, descriptions)
        break  # 仅展示一个批次，移除此行以展示所有批次