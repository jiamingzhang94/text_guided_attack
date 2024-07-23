import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
from pycocotools.coco import COCO
import clip
from models.ae_official import CLIPEncoder
from torch.nn.utils.rnn import pad_sequence
from lavis.datasets.builders import load_dataset
from lavis.tasks import RetrievalTask
import logging
import time
import torch.nn.functional as F
import datetime


class TransformDataset(Dataset):
    def __init__(self, base_dataset, root):
        self.base_dataset = base_dataset
        self.root = root
        self.text = base_dataset.text
        self.txt2img = base_dataset.txt2img
        self.img2txt = base_dataset.img2txt

        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset.image[idx]
        image_path = os.path.join(self.root, item)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image

class COCORetrievalDataset(Dataset):
    def __init__(self, root, annFile):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns]

        return image, captions, img_id

    def __len__(self):
        return len(self.ids)


def custom_collate_fn(batch):
    images, captions, img_ids = zip(*batch)
    images = torch.stack(images, dim=0)
    max_caption_length = max(len(caps) for caps in captions)
    padded_captions = [caps + [''] * (max_caption_length - len(caps)) for caps in captions]
    return images, padded_captions, img_ids


class EvaluatorCOCO:
    def __init__(self, model_name, batch_size, coco_root, ann_file, device):
        self.batch_size = batch_size
        self.coco_root = coco_root
        self.ann_file = ann_file
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.model = self.initialize_model(model_name)

    def initialize_model(self, model_name):
        model = CLIPEncoder(model_name).to(self.device)
        return model

    def create_coco_dataloader(self):
        coco_dataset = load_dataset("coco_retrieval",
                                    vis_path=self.coco_root
                                    )
        fixed_dataset = TransformDataset(coco_dataset["val"], self.coco_root)
        dataloader = DataLoader(fixed_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False, num_workers=16)
        # coco_dataset = COCORetrievalDataset(root=self.coco_root, annFile=self.ann_file)
        # dataloader = DataLoader(coco_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False, num_workers=16, collate_fn=custom_collate_fn)
        return dataloader

    # @torch.no_grad()
    # def extract_features(self, dataloader):
    #     image_features, text_features, all_captions = [], [], []
    #
    #     for images, captions, _ in tqdm(dataloader):
    #         images = images.to(self.device)
    #         image_feat = self.model.encode_img(images)
    #         image_features.append(image_feat)
    #
    #         batch_captions = [cap for cap_set in captions for cap in cap_set if cap]
    #         cap_features = self.model.encode_text(batch_captions)
    #         text_features.extend(cap_features.split([len(cap_set) for cap_set in captions]))
    #         all_captions.extend(captions)
    #
    #     return torch.cat(image_features), pad_sequence(text_features, batch_first=True), all_captions
    @torch.no_grad()
    def extract_features(self, dataloader):
        logging.info("Computing features for evaluation...")
        start_time = time.time()

        texts = dataloader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_features = []

        for i in range(0, num_text, text_bs):

            text = texts[i: min(num_text, i + text_bs)]
            # text_input = self.tokenizer(text).to(self.device)

            text_feat = self.model.encode_text(text)
            text_feat = F.normalize(text_feat, dim=-1)

            text_features.append(text_feat)

        text_features = torch.cat(text_features, dim=0)

        image_features = []
        for samples in dataloader:
            image = samples

            image = image.to(self.device)
            image_feat = self.model.encode_img(image)
            image_feat = F.normalize(image_feat, dim=-1)

            image_features.append(image_feat)

        image_features = torch.cat(image_features, dim=0)

        sims_matrix_i2t = image_features @ text_features.t()
        sims_matrix_t2i = sims_matrix_i2t.t()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))

        return sims_matrix_i2t.cpu().numpy(), sims_matrix_t2i.cpu().numpy()

    @staticmethod
    @torch.no_grad()
    def compute_metrics(scores_i2t, scores_t2i, txt2img, img2txt):

        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        agg_metrics = (tr1 + tr5 + tr10) / 3

        eval_result = {
            "txt_r1": tr1,
            "txt_r5": tr5,
            "txt_r10": tr10,
            "txt_r_mean": tr_mean,
            "img_r1": ir1,
            "img_r5": ir5,
            "img_r10": ir10,
            "img_r_mean": ir_mean,
            "r_mean": r_mean,
            "agg_metrics": agg_metrics,
        }
        return eval_result
    # @torch.no_grad()
    # def extract_features(self, dataloader):
    #     img_embs, cap_embs = [], []
    #     for batch in tqdm(dataloader):
    #         images = batch["image"].to(self.device)
    #         captions = batch["text_input"]
    #
    #         img_emb = self.model.encode_img(images)
    #         cap_emb = self.model.encode_text(captions)
    #
    #         img_embs.append(img_emb)
    #         cap_embs.append(cap_emb)
    #
    #     return torch.cat(img_embs), torch.cat(cap_embs)

    # @torch.no_grad()
    # def compute_metrics(self, image_features, text_features):
    #     similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    #     values, indices = similarity.topk(10, dim=-1)
    #
    #     def calculate_recall(indices, num_samples):
    #         recall = {k: 0.0 for k in [1, 5, 10]}
    #         for i in range(num_samples):
    #             for k in recall.keys():
    #                 if i in indices[:, :k].flatten():
    #                     recall[k] += 1
    #         return {k: v / num_samples for k, v in recall.items()}
    #
    #     i2t_recall = calculate_recall(indices, len(image_features))
    #     t2i_recall = calculate_recall(indices.T, len(text_features))
    #
    #     return i2t_recall, t2i_recall

    def eval(self):
        data_loader = self.create_coco_dataloader()

        print("Extracting features...")
        score_i2t, score_t2i = self.extract_features(data_loader)

        print("Computing metrics...")
        eval_result = self.compute_metrics(
            score_i2t,
            score_t2i,
            data_loader.dataset.txt2img,
            data_loader.dataset.img2txt,
        )
        logging.info(eval_result)
        print(eval_result)

        # print("Image to Text Recall:")
        # for k, v in i2t_recall.items():
        #     print(f"R@{k}: {v:.4f}")
        #
        # print("\nText to Image Recall:")
        # for k, v in t2i_recall.items():
        #     print(f"R@{k}: {v:.4f}")

# 使用示例
# evaluator = Evaluator(tar_dir='path_to_tar_files', batch_size=32, coco_root='path_to_coco_root', ann_file='path_to_ann_file', device='cuda')
# evaluator.eval()
