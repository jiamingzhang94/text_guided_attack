import argparse
import os
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TORCH_HOME'] = '/new_data/yifei2/junhong/text_guide_attack/cache'
import torch
import tqdm
from su import SU
from sasd_ws import SASD_WS
from utils import *
from torchvision.datasets import ImageFolder
import torch.nn.functional as F


def get_parser():

    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('--attack', default='su', type=str, help='the attack algorithm')
    parser.add_argument('--batch_size', default=25, type=int, help='the bacth size')
    parser.add_argument('--eps', default=8 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.0 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--model', default='ViT-B/32', type=str, help='the source surrogate model')
    # parser.add_argument('--model', default='resnet50', type=str, help='the source surrogate model')
    parser.add_argument('--loss_type', default='cos', type=str)
    parser.add_argument('--clean_image_path', default="/new_data/yifei2/junhong/AttackVLM-main/data/imagenet-1K",
                        type=str,
                        help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--target_data_path',
                        default='/new_data/yifei2/junhong/dataset/ms_coco/coco/annotation/coco_karpathy_val.json',
                        type=str, help='the path to store the adversarial data')
    parser.add_argument('--target_image_path', default="/new_data/yifei2/junhong/dataset/ms_coco/coco/images", type=str,
                        help='the path to store the adversarial image')
    parser.add_argument('--output_path', default='/new_data/yifei2/junhong/text_guide_attack/saved_result/temp',
                        type=str,
                        help='the path to store the adversarial patches')
    parser.add_argument('--checkpoint_path', default='/new_data/yifei2/junhong/text_guide_attack/cache/hub/checkpoints', type=str, help='the size of the adversarial patch')


    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--ensemble', action='store_true', help='enable ensemble attack')
    parser.add_argument('--random_start', default=False, type=bool, help='set random start')
    parser.add_argument('--targeted', default=True, action='store_true', help='targeted attack')
    parser.add_argument('--GPU_ID', default='0', type=str)
    return parser.parse_args()


def compute_cosine_similarity(clean_features, target_features):
    cosine_similarity = F.cosine_similarity(clean_features, target_features, dim=1)
    return cosine_similarity


def main():
    args = get_parser()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    # imagenet_data = ImageFolder(args.clean_image_path, transform=eval_transform)
    clean_data = ImageFolderWithPaths(args.clean_image_path)
    target_data = ImageTextDataset(args.target_data_path,args.target_image_path)
    data_loader_target = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=8)
    data_loader_clean = torch.utils.data.DataLoader(clean_data, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # dataset = AdvDataset(input_dir=args.input_dir, output_path=args.output_path, targeted=args.targeted, eval=args.eval)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)

    if args.ensemble or len(args.model.split(',')) > 1:
        args.model = args.model.split(',')
    if args.attack == 'su':
        attacker = SU(model_name=args.model, loss=args.loss_type, epsilon=args.eps,alpha=args.alpha,targeted=True)
    elif args.attack == 'sasd_ws':
        attacker = SASD_WS(model_name=args.model, loss=args.loss_type, epsilon=args.eps,alpha=args.alpha,targeted=True,checkpoint_path=args.checkpoint_path)
    else:
        raise Exception("Unsupported method {}".format(args.attack))
    # attacker = transferattack.load_attack_class(args.attack)(model_name=args.model, targeted=args.targeted)

    img_id = 0
    adv_data = []
    sim = []
    progress_bar = tqdm.tqdm(
        enumerate(zip(data_loader_clean, data_loader_target)),
        total=len(data_loader_target),
        desc=f"{args.attack}_{attacker.loss_name}"
    )

    for batch_idx, ((clean_image), (target_image, text)) in progress_bar:
        clean_image = clean_image.to(attacker.device)
        target_image = target_image.to(attacker.device)
        perturbations = attacker(clean_image, target_image)
        adv_image = clean_image + perturbations

        adv_image_path = args.output_path + "/adv_images"
        if not os.path.exists(adv_image_path):
            os.makedirs(adv_image_path)
        for i in range(adv_image.shape[0]):
            adversary=(adv_image[i].detach().permute((1, 2, 0)).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(adversary).save(os.path.join(adv_image_path, f"{img_id:05d}.png"))
            # save_images(adv_image_path, adv_image[i].unsqueeze(0), [f"{img_id:05d}.png"])
            # torchvision.utils.save_image(adv_image[i], os.path.join(adv_image_path, f"{img_id:05d}.") + 'png')
            adv_data.append(
                {
                    'image': f"{img_id:05d}.png",
                    'caption': [text[i]]
                }
            )
            img_id += 1

        clean_features = attacker.model(clean_image)
        adv_features = attacker.model(adv_image)
        similarity = compute_cosine_similarity(clean_features, adv_features)
        mean_similarity = similarity.mean().item()
        sim.append(mean_similarity)
        progress_bar.set_postfix(
            similartity=f'{mean_similarity:.4f}'
        )
    with open(args.output_path + "/adv_images.json", "w", encoding='utf-8') as f:
        json.dump(adv_data, f, indent=4, ensure_ascii=False)
    print(f"Total similarity : {sum(sim) / len(sim):.4f}")


if __name__ == '__main__':
    main()
