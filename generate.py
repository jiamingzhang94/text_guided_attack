import os
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import argparse
from models.decoder_gpt4o import Decoder
import torch.profiler
import torchvision
from models import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)  # (img, label)
        path, _ = self.samples[index]  # path: str
        # if index==0:
        #     print(original_tuple)
        #     print(original_tuple[0])
        # image_processed = vis_processors["eval"](original_tuple[0])
        # text_processed  = txt_processors["eval"](class_text_all[original_tuple[1]])
        # image_processed = preprocess(original_tuple[0]).to(device)
        image_processed = self.transform(original_tuple[0]).to(device)

        return image_processed, original_tuple[1], path


class ImageTextDataset(Dataset):
    def __init__(self, it_pair, image_path, image_only, transform=None):
        self.it_pair = it_pair
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
            image = Image.open(image_path).convert('RGB')
            # 应用转换
            if self.transform:
                image = self.transform(image)

            return image, _
        else:
            text = sample['text']
            return _, text


class Projection(nn.Module):
    def __init__(self):
        super(Projection, self).__init__()
        self.model = 0

    def forward(self, x):
        return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--model_name", type=str, default="ViT-B/16")
    parser.add_argument("--image_only", type=bool, default=True)
    parser.add_argument("--clean_image_path", type=str,
                        default="/new_data/yifei2/junhong/AttackVLM-main/data/imagenet-1K")
    parser.add_argument("--target_caption", type=str,
                        default="/new_data/yifei2/junhong/text_guide_attack/data/caption_5000.json")
    parser.add_argument("--target_image_path", type=str, default="/new_data/yifei2/junhong/dataset/COCO-2017/train2017")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_path", type=str, default="/new_data/yife")
    args = parser.parse_args()

    # model
    print(f"Loading CLIP models: {args.model_name}...")
    clip_model, preprocess = clip.load(args.model_name, device=device, jit=False,
                                       download_root="/new_data/yifei2/junhong/AttackVLM-main/model/clip")
    decoder = Decoder(embed_dim=512)
    projection = Projection()
    print("Done")

    # datalodaer
    imagenet_data = ImageFolderWithPaths(args.clean_image_path, transform=preprocess)
    target_data = ImageTextDataset(args.target_caption, args.target_image_path, args.image_only, transform=preprocess)

    data_loader_imagenet = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=0, drop_last=False)
    data_loader_target = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=0, drop_last=False)

    adv_tensor=[]
    for idx, ((clean_image, _, path), (target_image, text)) in enumerate(zip(data_loader_imagenet, data_loader_target)):
        clean_image = clean_image.to(device)
        target_image = target_image.to(device)

        if args.image_only:
            img_emb = clip_model.encode_image(target_image)
        else:
            text_input = clip.tokenize(text, truncate=True).to(device)
            text_emb = clip_model.encode_text(text_input)
            img_emb = projection(text_emb)

        noise = decoder(img_emb)
        noise = torch.clamp(noise, -args.eps, args.eps)
        adv_image = clean_image + noise
        if idx==0:
            adv_tensor= adv_image.unsqueeze(0)
        else:
            tmp_adv_image=adv_image.unsqueeze(0)
            adv_tensor= torch.stack((adv_tensor,tmp_adv_image),dim=0)

        for i in range(adv_image.shape[0]):
            torchvision.utils.save_image(adv_image[i], os.path.join(args.output, f"{idx:05d}.") + 'png')
    torch.save(args.output, adv_tensor)
