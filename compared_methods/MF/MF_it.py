import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TORCH_HOME'] = '/new_data/yifei2/junhong/AttackVLM-main/model/blip-cache'

# 获取当前文件的上两级目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# 将 models 目录添加到 sys.path
models_dir = os.path.join(project_root, 'models')
sys.path.append(models_dir)
import random
import clip
import numpy as np
import torch
import torchvision
import json
from PIL import Image

from lavis.models import load_model_and_preprocess

# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"


# basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# combine
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)


# ------------------------------------------------------------------ #

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)  # (img, label)
        path, _ = self.samples[index]  # path: str
        # if index==0:
        #     print(original_tuple)
        #     print(original_tuple[0])
        # image_processed = vis_processors["eval"](original_tuple[0])
        # text_processed  = txt_processors["eval"](class_text_all[original_tuple[1]])
        image_processed = preprocess(original_tuple[0]).to(device)

        return image_processed, original_tuple[1], path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--num_samples", default=20, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--output", default="/new_data/yifei2/junhong/text_guide_attack/saved_result/MF_it", type=str,
                        help='the folder name that restore your outputs')

    parser.add_argument("--model_name", default="ViT-B/32", type=str)
    parser.add_argument("--clean_image", type=str, default="/new_data/yifei2/junhong/AttackVLM-main/data/imagenet-1K")
    parser.add_argument("--target_caption", type=str,
                        default="/new_data/yifei2/junhong/dataset/ms_coco/coco/annotation/coco_karpathy_val.json")
    # parser.add_argument("--model_type", default="base_coco", type=str)
    args = parser.parse_args()

    alpha = args.alpha
    epsilon = args.epsilon

    # for normalized imgs
    scaling_tensor = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device)
    scaling_tensor = scaling_tensor.reshape((3, 1, 1)).unsqueeze(0)
    alpha = args.alpha / 255.0 / scaling_tensor
    epsilon = args.epsilon / 255.0 / scaling_tensor

    # select and load model

    print(f"Loading CLIP models: {args.model_name}...")
    clip_model, preprocess = clip.load(args.model_name, device=device, jit=False,
                                       download_root="/new_data/yifei2/junhong/AttackVLM-main/model/clip")
    print(f"Done")

    # ------------- pre-processing images/text ------------- #
    imagenet_data = ImageFolderWithPaths(args.clean_image, transform=None)
    # target_data = ImageFolderWithPaths(args.target_image, transform=None)

    data_loader_imagenet = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=0, drop_last=False)
    # data_loader_target = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=False,
    #                                                  num_workers=0, drop_last=False)
    inverse_normalize = torchvision.transforms.Normalize(
        mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
        std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])

    img_id = 0
    adv_data = []
    with open(args.target_caption, 'r', encoding='utf-8') as f:
        text_data = []
        target_json = json.load(f)
        for i in target_json:
            text_data.append(i['caption'][0].strip())
    # start attack
    for i, ((image_org, _, path), text) in enumerate(zip(data_loader_imagenet, text_data)):
        if args.batch_size * (i + 1) > args.num_samples:
            break

        # (bs, c, h, w)
        image_org = image_org.to(device)

        # extract text features
        with torch.no_grad():
            text = clip.tokenize(text).to(device)
            tgt_text_features = clip_model.encode_text(text)
            tgt_text_features = tgt_text_features / tgt_text_features.norm(dim=1, keepdim=True)

        # -------- get adv image -------- #
        delta = torch.zeros_like(image_org, requires_grad=True)
        for j in range(args.steps):
            adv_image = image_org + delta  # image is normalized to (0.0, 1.0)
            adv_image_features = clip_model.encode_image(adv_image)
            adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)

            embedding_sim = torch.mean(torch.sum(adv_image_features * tgt_text_features, dim=1))  # cos. sim
            embedding_sim.backward()

            grad = delta.grad.detach()
            delta_data = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            delta.data = delta_data
            delta.grad.zero_()
            print(
                f"iter {i}/{args.num_samples // args.batch_size} step:{j:3d}, embedding similarity={embedding_sim.item():.5f}, max delta={torch.max(torch.abs(delta_data)).item():.3f}, mean delta={torch.mean(torch.abs(delta_data)).item():.3f}")

        # save imgs
        adv_image = image_org + delta
        adv_image = torch.clamp(inverse_normalize(adv_image), 0.0, 1.0)

        adv_image_path = args.output + "/adv_images"
        if not os.path.exists(adv_image_path):
            os.makedirs(adv_image_path)
        for i in range(adv_image.shape[0]):
            torchvision.utils.save_image(adv_image[i], os.path.join(adv_image_path, f"{img_id:05d}.") + 'png')
            adv_data.append(
                {
                    'image': f"{img_id:05d}.png",
                    'caption': [text_data[img_id]]
                }
            )
            img_id += 1
    with open(args.output + "/adv_images.json", "w", encoding='utf-8') as f:
        json.dump(adv_data, f, indent=4, ensure_ascii=False)
