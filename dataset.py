import torch
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import requests
from io import BytesIO
import argparse
import os
class MS_COCO(Dataset):
    def __init__(self, args):
        self.args = args
        self.data_list=pq.read_table(args.data_path)
        self.data_list=self.data_list.to_pandas()
        self.image_path=args.image_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        text = self.data_list['TEXT'][idx]

        image_name = self.data_list['URL'][idx]

        img_path=os.path.join(self.image_path,image_name)
        img = Image.open(img_path).convert('RGB')

        # You can perform additional transformations on the image here if needed

        return img, text

def collate_fn(batch):
    images, texts = zip(*batch)
    images = list(images)
    # texts = list(texts)
    texts=[text.tolist() for text in texts]
    return images, texts



if __name__== '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--train", type=str, default=True)
    argparse.add_argument("--clip_model_path", type=str, default="/data2/ModelWarehouse/clip-vit-base-patch32")
    argparse.add_argument("--image_path", type=str, default="/data2/zhiyu/data/coco/images/train2017")
    argparse.add_argument("--data_path", type=str, default="/data2/junhong/proj/text_guide_attack/data/mscoco_exist.parquet")
    args = argparse.parse_args()
    ms_coco=MS_COCO(args)
    print(len(ms_coco))
    # Example DataLoader parameters
    batch_size = 32
    shuffle = True
    # Create DataLoader
    data_loader = DataLoader(ms_coco, batch_size=batch_size, shuffle=shuffle,collate_fn=collate_fn)
    # Example usage in training loop
    # print(len(ms_coco))
    # train2017=os.listdir("/data2/zhiyu/data/coco/images/train2017")
    # print(len(train2017))
    num_batches_to_iterate=5
    for batch_idx, (images, texts) in enumerate(data_loader):
        if batch_idx >= num_batches_to_iterate:
            break
        print(f"Batch {batch_idx}, Number of samples: {len(images)}")
        print(f"Text: {texts[0]}")  # Print text description
    # print(ms_coco.__getitem__(98397))
