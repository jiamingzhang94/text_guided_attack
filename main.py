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
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from models.ae_official import CLIPEncoder
from models.decoder_gpt4o import *

from torch.utils.tensorboard import SummaryWriter
import torch.profiler

from eval_ls import *


class BatchContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, temperature=0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, embeddings1, embeddings2):
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        assert embeddings1.size() == embeddings2.size(), "Embeddings dimensions don't match!"
        batch_size = embeddings1.size(0)

        similarities = torch.mm(embeddings1, embeddings2.t()) / self.temperature

        labels = torch.eye(batch_size, device=similarities.device)

        pos_loss = -torch.sum(labels * F.log_softmax(similarities, dim=1)) / batch_size
        neg_loss = -torch.sum((1 - labels) * F.log_softmax(-similarities + self.margin, dim=1)) / batch_size

        loss = pos_loss + neg_loss

        return loss


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings1, embeddings2):
        # Normalize the embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature
        # Labels are all diagonal elements (where i == j)
        labels = torch.arange(similarity_matrix.shape[0], device=similarity_matrix.device)
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


def main(args):
    tar_files = glob.glob(os.path.join(args.tar_dir, "*.tar"))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    def handle_sample(sample):
        image, text = sample
        return train_transform(image), text

    dataset = (
        wds.WebDataset(tar_files)
           .decode("pil")
           .to_tuple("jpg", "txt")
           .map(handle_sample)
           .batched(args.batch_size)
           .shuffle(10000)
    )

    train_data_loader = DataLoader(dataset, batch_size=None, num_workers=16, pin_memory=True, persistent_workers=True)

    clip_encoder = CLIPEncoder('ViT-B/32').to(args.device)
    decoder = Decoder(embed_dim=512).to(args.device)

    for param in clip_encoder.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(decoder.parameters(), lr=1e-4)
    # optimizer = optim.SGD(decoder.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10000, T_mult=2)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    # warmup_scheduler = LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler()
    writer = SummaryWriter(log_dir='./runs')
    criterion = InfoNCELoss()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as profiler:
        global_step = 0
        for epoch in range(args.epoch):
            total_loss = 0
            count = 0
            for batch_idx, (images, text) in enumerate(train_data_loader):
                images = images.to(args.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    e1 = clip_encoder.encode_img(images)
                with autocast():
                    noise = decoder(e1)
                    noise = torch.clamp(noise, -args.eps, args.eps)
                    images_adv = []
                    for _ in range(args.chunk):
                        images_adv.append(torch.clamp(noise + images[torch.randperm(images.size(0))], 0, 1))
                    images_adv = torch.cat(images_adv, dim=0)
                    e2 = clip_encoder.encode_img(images_adv)
                    e2_chunks = torch.chunk(e2, args.chunk, dim=0)
                    sum_tensor = torch.zeros_like(e2_chunks[0])
                    for chunk in e2_chunks:
                        sum_tensor += chunk
                    e2 = sum_tensor / args.chunk

                    loss = criterion(e1, e2)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # loss = criterion(e1, e2)
                # loss.backward()
                # optimizer.step()

                total_loss += loss.item()
                count += images.size(0)
                global_step += 1
                scheduler.step()
                profiler.step()

                if batch_idx % 500 == 0:
                    avg_loss = total_loss / count
                    print(f'Batch {batch_idx}, Loss: {total_loss/count:.4f}')
                    writer.add_scalar('Loss/train', avg_loss, global_step)

            avg_loss = total_loss / count
            print(f"Training Loss: {avg_loss}")
            writer.add_scalar('Loss/epoch', avg_loss, epoch)

            torch.save({
                'epoch': epoch,
                'decoder_state_dict': decoder.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f"checkpoints/model_epoch_{epoch}.pt")
    writer.close()


def eval(args):
    evaluator = EvaluatorCOCO(batch_size=args.batch_size,
                              model_name='ViT-B/32',
                              coco_root='/home/dycpu6_8tssd1/jmzhang/datasets/mscoco',
                              ann_file='/home/dycpu6_8tssd1/jmzhang/datasets/mscoco/captions_val2014.json',
                              device=args.device)
    evaluator.eval()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--tar_dir", type=str, default="/home/dycpu6_8tssd1/jmzhang/datasets/coco/mscoco")
    # parser.add_argument("--tar_dir", type=str, default="/home/dycpu4_data1/jmzhang/big_datasets/laion-400m/laion400m-data")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--auto_cast", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--chunk", type=int, default=5)
    parser.add_argument("--eps", type=float, default=8/255)
    args = parser.parse_args()

    # main(args)
    eval(args)