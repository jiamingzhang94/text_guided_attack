import os
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from PIL import Image
from torch.utils.data import Dataset
import argparse
from models.decoder_gpt4o import Decoder
import torch.profiler
import torchvision
from models import clip
import json
from torch.nn.functional import cosine_similarity
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from models.ae_official import CLIPEncoder
from projection import ProjectionNetwork


# class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
#     def __getitem__(self, index: int):
#         original_tuple = super().__getitem__(index)  # (img, label)
#         path, _ = self.samples[index]  # path: str
#         # original_tuple[0]=original_tuple[0].convert('RGB')
#         # if index==0:
#         #     print(original_tuple)
#         #     print(original_tuple[0])
#         # image_processed = vis_processors["eval"](original_tuple[0])
#         # text_processed  = txt_processors["eval"](class_text_all[original_tuple[1]])
#         # image_processed = preprocess(original_tuple[0]).to(device)
#         image_processed = preprocess(original_tuple[0]).to(device)
#
#         return image_processed, original_tuple[1], path

def compute_cosine_similarity(image_features, text_features):
    assert image_features.shape == text_features.shape
    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)
    cosine_similarity = F.cosine_similarity(image_features, text_features, dim=1)
    return cosine_similarity


class ImageTextDataset(Dataset):
    def __init__(self, it_pair_path, image_path, image_only, transform=None):
        with open(it_pair_path, 'r', encoding='utf-8') as f:
            self.it_pair = json.load(f)
        self.transform = transform
        self.image_path = image_path
        self.image_only = image_only

    def __len__(self):
        return len(self.it_pair)

    def __getitem__(self, idx):
        sample = self.it_pair[idx]

        # 加载图像
        if self.image_only:
            image_path = os.path.join(self.image_path, sample['image'])
            # image = Image.open(image_path).convert('RGB')
            image = Image.open(image_path).convert('RGB')
            # 应用转换
            if self.transform:
                image = self.transform(image)
            text = sample['caption'][0]
            return image, text
        else:
            image = sample['image']
            text = sample['caption'][0]
            return image, text


class Projection(nn.Module):
    def __init__(self):
        super(Projection, self).__init__()
        self.model = 0

    def forward(self, x):
        return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=16 / 255)
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--decoder_path", type=str,
                        default="checkpoints/model_current.pt")
    parser.add_argument("--projection_path", type=str, default="")
    parser.add_argument("--image_only", type=bool, default=True)
    parser.add_argument("--clean_image_path", type=str,
                        default="/home/dycpu6_8tssd1/jmzhang/datasets/ILSVRC2012/val/")
    parser.add_argument("--target_caption", type=str,
                        default="json/coco_karpathy_val.json")
    parser.add_argument("--target_image_path", type=str,
                        default="/home/dycpu6_8tssd1/jmzhang/datasets/mscoco")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--output_path", type=str,
                        default="outputs")
    args = parser.parse_args()

    device = args.device
    # model
    print(f"Loading CLIP models: {args.model_name}...")
    # clip_model, preprocess = clip.load(args.model_name, device=device, jit=False,
    #                                    download_root="/new_data/yifei2/junhong/AttackVLM-main/model/clip")
    clip_model = CLIPEncoder('ViT-B/32').to(device)
    print(f"Loading Decoder: {args.decoder_path.split('/')[-1]}...")
    decoder = Decoder(embed_dim=512).to(device).eval()
    decoder.load_state_dict(torch.load(args.decoder_path, map_location='cpu')["decoder_state_dict"])

    if not args.image_only:
        print(f"Loading Projection: {args.projection_path.split('/')[-1]}...")
        projection = ProjectionNetwork().to(device).eval()
        projection.load_state_dict(torch.load(args.projection_path, map_location='cpu'))
        print("Done")

    # dataloader
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    # imagenet_data = ImageFolderWithPaths(args.clean_image_path, transform=preprocess)
    imagenet_data = ImageFolder(args.clean_image_path, transform=eval_transform)
    target_data = ImageTextDataset(args.target_caption, args.target_image_path, args.image_only, transform=eval_transform)

    data_loader_imagenet = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=8)
    data_loader_target = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=8)

    # inverse_normalize = torchvision.transforms.Normalize(
    #     mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
    #     std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])
    adv_tensor = []
    img_id = 0
    adv_data = []

    total_sim_emb = 0
    total_batches = 0

    for idx, ((clean_image, label), (target_image, text)) in tqdm.tqdm(enumerate(zip(data_loader_imagenet, data_loader_target))):
        clean_image = clean_image.to(device)
        target_image = target_image.to(device)

        # 得到对抗图像
        with torch.no_grad():
            if args.image_only:
                img_emb = clip_model.encode_img(target_image)
            else:
                # text_input = clip.tokenize(text, truncate=True).to(device)
                text_emb = clip_model.encode_text(text)
                img_emb = projection(text_emb)
            origin_noise = decoder(img_emb)
            noise = torch.clamp(origin_noise, -args.eps, args.eps)
            adv_image = clean_image + noise
            adv_image = torch.clamp(adv_image, 0, 1)
            # if idx == 0:
            #     adv_tensor = adv_image
            # else:
            #     adv_tensor = torch.cat((adv_tensor, adv_image), dim=0)

            # 计算对抗图像的clip embedding的相似度
            adv_emb = clip_model.encode_img(adv_image)
            sim_emb = compute_cosine_similarity(img_emb, adv_emb).mean()
            total_sim_emb += sim_emb.item()
            total_batches += 1
            # print(f"iter {idx}/{5000 // args.batch_size} clip_emb_similarity={sim_emb.item():.5f}")

        # save images
        # adv_image = torch.clamp(inverse_normalize(adv_image), 0.0, 1.0)
        adv_image_path = args.output_path + "/adv_images"
        if not os.path.exists(adv_image_path):
            os.makedirs(adv_image_path)
        for i in range(adv_image.shape[0]):
            torchvision.utils.save_image(adv_image[i], os.path.join(adv_image_path, f"{img_id:05d}.") + 'png')
            adv_data.append(
                {
                    'image': f"{img_id:05d}.png",
                    'caption':[text[i]]
                }
            )
            img_id += 1

    average_sim_emb = total_sim_emb / total_batches
    print(f"Average clip_emb_similarity for the entire epoch: {average_sim_emb:.5f}")
    # pt_path = args.output_path + "/adv_images.pt"
    # torch.save(adv_tensor, pt_path)
    with open(args.output_path + "/adv_images.json", "w",encoding='utf-8')as f:
        json.dump(adv_data,f,indent=4,ensure_ascii=False)
