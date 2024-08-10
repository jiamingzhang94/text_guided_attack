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
import lavis
from lavis.models import load_model_and_preprocess
# from lavis.datasets import load_dataset, DatasetBuilder
# from lavis.tasks import RetrievalTask


# model, vis_processors, text_processors = load_model_and_preprocess(
#     name="clip", model_type="ViT-B/32", is_eval=True
# )

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

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
#
# image = preprocess(Image.open("/homes/jmzhang/000000000.jpg")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
#
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     print(1)




import torch
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter
from dalle2_pytorch.trainer import DiffusionPriorTrainer


device = 'cpu'

def load_diffusion_model(dprior_path):

    prior_network = DiffusionPriorNetwork(
        dim=768,
        depth=24,
        dim_head=64,
        heads=32,
        normformer=True,
        attn_dropout=5e-2,
        ff_dropout=5e-2,
        num_time_embeds=1,
        num_image_embeds=1,
        num_text_embeds=1,
        num_timesteps=1000,
        ff_mult=4
    )

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=OpenAIClipAdapter("ViT-B/32"),
        image_embed_dim=768,
        timesteps=1000,
        cond_drop_prob=0.1,
        loss_type="l2",
        condition_on_text_encodings=True,

    )

    trainer = DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=1.1e-4,
        wd=6.02e-2,
        max_grad_norm=0.5,
        amp=False,
        group_wd_params=True,
        use_ema=True,
        device=device,
        accelerator=None,
    )

    trainer.load(dprior_path)

    return trainer

# 加载预训练的 DiffusionPrior 模型
dprior_path = 'C:/Users/admin/Downloads/ema472M.pth'
trainer = load_diffusion_model(dprior_path)
# tokenize the text
tokenized_text = clip.tokenize("<your amazing prompt>")
# predict an embedding
predicted_embedding = prior.sample(tokenized_text, n_samples_per_batch=2, cond_scale=1.0)