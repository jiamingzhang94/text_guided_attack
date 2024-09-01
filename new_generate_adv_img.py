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
from models.project import ProjectionNetwork
from dataset import *


class ImprovedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ImprovedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        cos_sim = torch.matmul(image_features, text_features.t()) / self.temperature
        labels = torch.eye(cos_sim.size(0), device=cos_sim.device)

        loss_i2t = -torch.sum(labels * F.log_softmax(cos_sim, dim=1), dim=1).mean()
        loss_t2i = -torch.sum(labels * F.log_softmax(cos_sim.t(), dim=1), dim=1).mean()

        loss = (loss_i2t + loss_t2i) / 2

        avg_similarity = torch.diag(torch.matmul(image_features, text_features.t())).mean()

        return loss, avg_similarity

criterion = ImprovedContrastiveLoss()
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

def process_save_data(task_type, output_path,origin_path,result_data,dataset):
    with open(origin_path,"r",encoding='utf-8') as f:
        origin_data=json.load(f)
    if task_type=='retrieval':
        retrieval_path=os.path.join(output_path,f'{dataset}.json')
        with open(retrieval_path,"w",encoding='utf-8') as f:
            json.dump(result_data,f,ensure_ascii=False,indent=4)

    elif task_type=='ve':
        for i,j in zip(origin_data,result_data):
            j["sentence"]=i["sentence"]
            j["label"]=i["label"]
        ve_path=os.path.join(output_path,f'{dataset}.json')
        with open(ve_path,"w",encoding='utf-8') as f:
            json.dump(result_data,f,ensure_ascii=False,indent=4)

    elif task_type=='caption':
        caption_path=os.path.join(output_path,f'{dataset}.json')
        caption_data=[]
        for i,j in zip(origin_data,result_data):
            i["image"]=j["image"]
            i["caption"]=j["caption"]
            caption_data.append(i)
        with open(caption_path,"w",encoding='utf-8') as f:
            json.dump(caption_data,f,ensure_ascii=False,indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=16./255)
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--decoder_path", type=str,
                        default="checkpoints/best_model_snli_ve_ImprovedContrastiveLoss.pt")
                        # default="checkpoints/model_mse.pt")

    parser.add_argument("--image_only", action="store_true", help="whether to use image only")
    parser.add_argument("--is_nocap", action="store_true", help="whether to use no caption")

    parser.add_argument("--clean_image_path", type=str,
                        default="/home/dycpu6_8tssd1/jmzhang/datasets/ILSVRC2012/val/")
    parser.add_argument("--target_caption", type=str,
                        default="json/ve_test_adv_1000.json",
                        help='/home/dycpu6_8tssd1/jmzhang/datasets/lavis/flickr30k/annotations/test.json '
                             'json/coco_karpathy_val_0_1000.json '
                             'json/ve_test_adv_1000.json')
    parser.add_argument("--target_image_path", type=str,
                        default="/home/dycpu6_8tssd1/jmzhang/datasets/lavis/flickr30k/images/flickr30k-images",
                        help='/home/dycpu6_8tssd1/jmzhang/datasets/lavis/flickr30k/images /home/dycpu6_8tssd1/jmzhang/datasets/mscoco')
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_path", type=str,
                        default="outputs")
    parser.add_argument("--adv_imgs", type=str, default="debug")
    parser.add_argument("--dataset", type=str, default="snli_ve", help='coco flickr30k')

    parser.add_argument("--task_type",type=str, default="retireval",choices=['retireval','ve','caption'])
    args = parser.parse_args()

    device = args.device
    # model
    print(f"Loading CLIP models: {args.model_name}...")
    # clip_model, preprocess = clip.load(args.model_name, device=device, jit=False,
    #                                    download_root="/new_data/yifei2/junhong/AttackVLM-main/model/clip")
    clip_model = CLIPEncoder('ViT-B/32').to(device)
    print(f"Loading Decoder: {args.decoder_path.split('/')[-1]}...")
    decoder = Decoder(embed_dim=512).to(device).eval()
    try:
        decoder.load_state_dict(torch.load(args.decoder_path, map_location='cpu')["decoder_state_dict"])
    except:
        state_dict = torch.load(args.decoder_path, map_location='cpu')["decoder_state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 去掉 'module.' 前缀
            new_state_dict[name] = v
        decoder.load_state_dict(new_state_dict)

    if not args.image_only:
        print(f"Loading Projection: {args.projection_path.split('/')[-1]}...")
        projection = ProjectionNetwork().to(device).eval()
        projection.load_state_dict(torch.load(args.projection_path, map_location='cpu'))

    # dataloader
    # imagenet_data = ImageFolderWithPaths(args.clean_image_path, transform=preprocess)
    # imagenet_data = SubsetImageFolder(args.clean_image_path, transform=eval_transform)
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

            # text_input = clip.tokenize(text, truncate=True).to(device)
            img_emb = clip_model.encode_img(target_image)
            target_emb = img_emb
            # img_emb = text_emb
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
            sim_emb = compute_cosine_similarity(target_emb, adv_emb).mean()
            loss, obser = criterion(target_emb, adv_emb)
            total_sim_emb += sim_emb.item()
            total_batches += 1
            # print(f"iter {idx}/{5000 // args.batch_size} clip_emb_similarity={sim_emb.item():.5f}")

        # save images
        # adv_image = torch.clamp(inverse_normalize(adv_image), 0.0, 1.0)
        adv_image_path = os.path.join(args.output_path, args.dataset, args.adv_imgs)
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
    print(f"Saved adv images: {args.dataset}/{args.adv_imgs}")
    # pt_path = args.output_path + "/adv_images.pt"
    # torch.save(adv_tensor, pt_path)
    # with open(args.output_path + f"/{args.dataset}_adv.json", "w",encoding='utf-8')as f:
    #     json.dump(adv_data,f,indent=4,ensure_ascii=False)
    process_save_data(args.task_type, args.output_path,args.target_caption,adv_data,args.dataset)
