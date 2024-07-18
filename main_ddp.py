import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
import glob
import webdataset as wds
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models.ae_official import CLIPEncoder
from models.decoder_gpt4o import Decoder
from torch.utils.tensorboard import SummaryWriter
import torch.profiler
from collections import deque
import time


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


# class InfoNCELoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.temperature = temperature
#
#     def forward(self, embeddings1, embeddings2):
#         embeddings1 = F.normalize(embeddings1, p=2, dim=1)
#         embeddings2 = F.normalize(embeddings2, p=2, dim=1)
#
#         similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature
#         labels = torch.arange(similarity_matrix.shape[0], device=similarity_matrix.device)
#         loss = F.cross_entropy(similarity_matrix, labels)
#
#         return loss

class DynamicInfoNCELoss(nn.Module):
    def __init__(self, initial_temp=1, final_temp=0.07, epochs=10):
        super().__init__()
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.epochs = epochs
        self.current_temp = initial_temp

    def forward(self, embeddings1, embeddings2):
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.current_temp
        labels = torch.arange(similarity_matrix.shape[0], device=similarity_matrix.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    def update_temperature(self, epoch):
        if epoch < self.epochs:
            self.current_temp = self.initial_temp - (self.initial_temp - self.final_temp) * (epoch / self.epochs)
        else:
            self.current_temp = self.final_temp


def get_lr(epoch, warmup_epochs=1):
    if epoch < warmup_epochs:
        current_lr = lr * (epoch + 1) / warmup_epochs
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = current_lr
        return lr * (epoch + 1) / warmup_epochs



def enumerate_report(seq, delta, growth=1.0):
    last = 0
    count = 0
    for count, item in enumerate(seq):
        now = time.time()
        if now - last > delta:
            last = now
            yield count, item, True
        else:
            yield count, item, False
        delta *= growth


def make_dataloader(tar_dir, batch_size, world_size, rank):
    """Create a DataLoader for training with WebDataset."""
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ])

    def handle_sample(sample):
        image, text = sample
        return transform(image), text

    tar_files = glob.glob(os.path.join(tar_dir, "*.tar"))
    dataset = (wds.WebDataset(tar_files, resampled=True, shardshuffle=True)
               .shuffle(5000)
               .decode("pil")
               .to_tuple("jpg", "txt")
               .map(handle_sample)
               .batched(batch_size)
               .with_epoch(len(tar_files))
               # .prepare(5)
               # .cache(size=5000)
               )

    loader = wds.WebLoader(dataset, batch_size=None, num_workers=16, pin_memory=True)
    # loader = loader.unbatched().shuffle(1000).batched(batch_size)
    return loader


def main_worker(args):
    args.local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend='gloo', init_method=args.dist_url, world_size=args.world_size, rank=args.local_rank)

    torch.cuda.set_device(args.local_rank)
    clip_encoder = CLIPEncoder('ViT-B/32').to(args.local_rank)
    decoder = Decoder(embed_dim=512).to(args.local_rank)

    clip_encoder = DistributedDataParallel(clip_encoder, device_ids=[args.local_rank])
    decoder = DistributedDataParallel(decoder, device_ids=[args.local_rank])

    for param in clip_encoder.parameters():
        param.requires_grad = False

    # optimizer = optim.Lamb(decoder.parameters(),
    #                        lr=1e-4,
    #                        betas=(0.9, 0.999))
    optimizer = torch.optim.AdamW(decoder.parameters(), 3e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=1)
    scaler = GradScaler()
    writer = SummaryWriter(log_dir='./runs') if args.local_rank == 0 else None
    criterion = DynamicInfoNCELoss()

    train_loader = make_dataloader(args.tar_dir, args.batch_size, args.world_size, args.local_rank)

    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir') if args.local_rank == 0 else None,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as profiler:
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=f'cuda:{args.local_rank}')
            decoder.module.load_state_dict(checkpoint['decoder_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            global_step = checkpoint.get('global_step', 0)
            start_epoch = checkpoint.get('epoch', 0) + 1
        else:
            global_step = 0
            start_epoch = 0
        for epoch in range(start_epoch, args.epoch):

            total_loss = 0
            count = 0
            criterion.update_temperature(epoch)
            for batch_idx, (images, text) in enumerate(train_loader):
                images = images.to(args.local_rank)
                optimizer.zero_grad()
                with torch.no_grad():
                    e1 = clip_encoder.module.encode_img(images)
                with autocast():
                    noise = decoder(e1)
                    noise = torch.clamp(noise, -args.eps, args.eps)
                    images_adv = []
                    for _ in range(args.chunk):
                        images_adv.append(torch.clamp(noise + images[torch.randperm(images.size(0))], 0, 1))
                    images_adv = torch.cat(images_adv, dim=0)
                    e2 = clip_encoder.module.encode_img(images_adv)
                    e2_chunks = torch.chunk(e2, args.chunk, dim=0)
                    sum_tensor = torch.zeros_like(e2_chunks[0])
                    for chunk in e2_chunks:
                        sum_tensor += chunk
                    e2 = sum_tensor / args.chunk

                    loss = criterion(e1, e2)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                count += images.size(0)
                global_step += 1
                scheduler.step()
                profiler.step()

                if batch_idx % 100 == 0 and args.local_rank == 0:
                    avg_loss = total_loss / count
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f'Batch {batch_idx}, Loss: {total_loss / count:.4f}, lr: {current_lr}')
                    torch.save({
                        'global_step': global_step,
                        'decoder_state_dict': decoder.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, f"checkpoints/model_current.pt")
                    writer.add_scalar('Loss/train', avg_loss, global_step)

            avg_loss = total_loss / count
            print(f"Epoch: {epoch}, Training Loss: {avg_loss}")
            if args.local_rank == 0:
                writer.add_scalar('Loss/epoch', avg_loss, epoch)
                torch.save({
                    'epoch': epoch,
                    'decoder_state_dict': decoder.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, f"checkpoints/model_epoch_{epoch}.pt")
    if args.local_rank == 0:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_dir", type=str, default="/home/dycpu6_8tssd1/jmzhang/datasets/coco/mscoco")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dist_url", type=str, default="tcp://127.0.0.1:23456")
    parser.add_argument("--chunk", type=int, default=5)
    parser.add_argument("--eps", type=float, default=16 / 255)
    parser.add_argument("--checkpoint", type=str, default=None, help="path to checkpoint to load")
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    main_worker(args)
