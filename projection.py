import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

from torchvision import transforms
import glob
import webdataset as wds
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from models.ae_official import CLIPEncoder
from models.decoder_gpt4o import Decoder
from lavis.common.config import Config
from lavis.datasets.builders import load_dataset
from PIL import Image
from torch.utils.data import Dataset





class EvalDataset(Dataset):
    def __init__(self, data_list, image_dir):
        self.data_list = data_list
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet标准化
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # 加载图像
        image_path = os.path.join(self.image_dir, item['image'])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 获取caption
        captions = item['caption']

        return {
            'image': image,
            'captions': captions
        }


def compute_cosine_similarity(image_features, text_features):
    assert image_features.shape == text_features.shape
    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)
    cosine_similarity = F.cosine_similarity(image_features, text_features, dim=1)
    return cosine_similarity


class ProjectionNetwork(nn.Module):
    def __init__(self, dim=512):
        super(ProjectionNetwork, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        residual = x
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x + residual


def train_one_epoch(args, train_data_loader, optimizer, clip_encoder, projection_network, criterion, scheduler, epoch):
    projection_network.train()
    total_loss = 0
    count = 0
    total_original_similarity = 0
    total_projected_similarity = 0
    for batch_idx, batch in enumerate(train_data_loader):
        images = batch['image'].to(args.device)
        captions = batch['caption']
        optimizer.zero_grad()
        with torch.no_grad():
            image_features = clip_encoder.encode_img(images)
            text_features = clip_encoder.encode_text(captions)
            image_features = image_features.float()
            text_features = text_features.float()
        projected_features = projection_network(text_features)
        loss = criterion(projected_features, image_features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            original_similarity = compute_cosine_similarity(image_features, text_features)
            projected_similarity = compute_cosine_similarity(image_features, projected_features)
            total_original_similarity += original_similarity.sum().item()
            total_projected_similarity += projected_similarity.sum().item()

        count += images.size(0)

        # if batch_idx % 100 == 0:
        #     avg_loss = total_loss / count
        #     print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss:.4f}')

    avg_loss = total_loss / count
    avg_original_similarity = total_original_similarity / count
    avg_projected_similarity = total_projected_similarity / count
    print(f"Epoch {epoch}, Training Loss: {avg_loss}")
    print(f"Epoch Average Original Similarity: {avg_original_similarity:.4f}")
    print(f"Epoch Average Projected Similarity: {avg_projected_similarity:.4f}")


def eval_one_epoch(args, val_data_loader, clip_encoder, projection_network, criterion):
    projection_network.eval()
    total_loss = 0
    count = 0
    total_original_similarity = 0
    total_projected_similarity = 0
    for batch_idx, batch in enumerate(val_data_loader):
        images = batch['images'].to(args.device)
        captions = batch['captions']
        caption_lengths = batch['caption_lengths']
        with torch.no_grad():
            flattened_captions = [caption for sublist in captions for caption in sublist]
            expanded_images = []
            for img, repeat_count in zip(images, caption_lengths):
                expanded_images.extend([img] * repeat_count)
            expanded_images = torch.stack(expanded_images)
            assert len(flattened_captions) == expanded_images.size(0), "Caption 数量与扩展后的图像数量不匹配"
            image_features = clip_encoder.encode_img(expanded_images)
            text_features = clip_encoder.encode_text(flattened_captions)

            image_features = image_features.float()
            text_features = text_features.float()
        projected_features = projection_network(text_features)
        loss = criterion(projected_features, image_features)

        total_loss += loss.item()
        with torch.no_grad():
            original_similarity = compute_cosine_similarity(image_features, text_features)
            projected_similarity = compute_cosine_similarity(image_features, projected_features)
            total_original_similarity += original_similarity.sum().item()
            total_projected_similarity += projected_similarity.sum().item()
        count += len(flattened_captions)

    avg_loss = total_loss / count
    avg_original_similarity = total_original_similarity / count
    avg_projected_similarity = total_projected_similarity / count
    print(f"Eval Loss: {avg_loss}")
    print(f"Average Original Similarity: {avg_original_similarity:.4f}")
    print(f"Average Projected Similarity: {avg_projected_similarity:.4f}")
    print('----------------------------------------------')
    return avg_projected_similarity, avg_loss


def train_collate_fn(batch):
    # 定义图像转换
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet标准化
    ])

    images = []
    captions = []
    for item in batch:
        image = item['image']
        caption = item['text_input']

        # 转换图像
        if isinstance(image, Image.Image):
            image = transform(image)

        images.append(image)
        captions.append(caption)

    # 将图像堆叠成一个批次
    images = torch.stack(images)

    return {
        'image': images,
        'caption': captions
    }


def eval_collate_fn(batch):
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet标准化
    ])

    images = []
    captions = []
    for item in batch:
        image = item['image']
        caption = item['text_input']

        # 转换图像
        if isinstance(image, Image.Image):
            image = transform(image)

        images.append(image)
        captions.append(caption)

    # 将图像堆叠成一个批次
    images = torch.stack(images)

    return {
        'image': images,
        'caption': captions
    }


def custom_collate(batch):
    images = torch.stack([item['image'] for item in batch])
    captions = [item['captions'] for item in batch]

    # 获取每个样本的caption数量
    caption_lengths = [len(c) for c in captions]

    return {
        'images': images,
        'captions': captions,
        'caption_lengths': caption_lengths
    }

def main():
    args = parse_args()
    coco_dataset = load_dataset("coco_retrieval", vis_path='/home/dycpu6_8tssd1/jmzhang/datasets/mscoco')
    train_dataset = coco_dataset['train']
    val_dataset = coco_dataset['val']
    test_dataset = coco_dataset['test']
    val_dataset = EvalDataset(val_dataset.annotation, image_dir='/home/dycpu6_8tssd1/jmzhang/datasets/mscoco')

    train_data_loader = DataLoader(train_dataset, batch_size=256, num_workers=8, pin_memory=True, shuffle=True,
                                   collate_fn=train_collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=800, num_workers=8, pin_memory=True, shuffle=False,
                                 collate_fn=custom_collate)

    clip_encoder = CLIPEncoder('ViT-B/32', args.device).to(args.device)
    decoder = Decoder(embed_dim=512).to(args.device)
    decoder.load_state_dict(torch.load('checkpoints/model_current.pt', map_location='cpu')["decoder_state_dict"])
    projection_network = ProjectionNetwork().to(args.device)
    clip_encoder.eval()  # Ensure CLIP encoder is in evaluation mode

    for param in clip_encoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(projection_network.parameters(), lr=1e-4, weight_decay=1e-5)
    # optimizer = optim.SGD(decoder.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10000, T_mult=2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    # warmup_scheduler = LambdaLR(optimizer, lr_lambda)

    criterion = nn.MSELoss()
    similarity = 0
    for epoch in range(args.num_epoch):
        train_one_epoch(args=args, train_data_loader=train_data_loader, optimizer=optimizer,
                        clip_encoder=clip_encoder, projection_network=projection_network, criterion=criterion,
                        scheduler=scheduler, epoch=epoch)
        current_similarity, avg_loss = eval_one_epoch(args=args, val_data_loader=val_data_loader,
                                                      clip_encoder=clip_encoder, projection_network=projection_network,
                                                      criterion=criterion)
        scheduler.step(avg_loss)
        if current_similarity > similarity:
            similarity = current_similarity
            torch.save(projection_network.state_dict(), 'checkpoints/projection.pt')

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--device", default='cuda:5')
    parser.add_argument("--num_epoch", default=120, type=int)
    parser.add_argument("--cfg_path", default="lavis_tool/clip/ret_coco_eval.yaml", help="path to configuration file.")
    parser.add_argument("--cache_path", default="/home/dycpu6_8tssd1/jmzhang/datasets", help="path to dataset cache")
    parser.add_argument("--data_path", help="test data path")
    parser.add_argument("--image_path", default='/home/dycpu6_8tssd1/jmzhang/datasets/mscoco',
                        help="path to image dataset")
    # parser.add_argument("--image_path",default="/new_data/yifei2/junhong/dataset/new_coco/coco/images",help="path to image dataset")
    parser.add_argument("--output_dir", help="path where to save result")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


if __name__ == "__main__":
    main()
