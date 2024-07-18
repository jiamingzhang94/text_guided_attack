import torch
from torch.utils.data import Dataset, DataLoader
import webdataset as wds
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
from pycocotools.coco import COCO
import clip
from models.ae_official import CLIPEncoder



class COCORetrievalDataset(Dataset):
    def __init__(self, root, annFile):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
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
        self.model = self.initialize_model(model_name)

    def initialize_model(self, model_name):
        model = CLIPEncoder(model_name).to(self.device)
        return model

    def create_coco_dataloader(self):
        coco_dataset = COCORetrievalDataset(root=self.coco_root, annFile=self.ann_file)
        return DataLoader(coco_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False, num_workers=16, collate_fn=custom_collate_fn)


    def extract_features(self, dataloader):
        image_features, text_features, all_captions = [], [], []

        for images, captions, _ in tqdm(dataloader):
            images = images.to(self.device)
            image_feat = self.model.encode_img(images)
            image_features.append(image_feat)

            for cap_set in captions:
                valid_caps = [cap for cap in cap_set if cap]  # 过滤掉空字符串
                cap_features = [self.model.encode_text([cap]) for cap in valid_caps]
                text_features.append(torch.cat(cap_features).mean(dim=0, keepdim=True))
                all_captions.append(valid_caps)

        return torch.cat(image_features), torch.cat(text_features), all_captions

    @torch.no_grad()
    def compute_metrics(self, image_features, text_features):
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity.topk(10, dim=-1)

        def calculate_recall(indices, num_samples):
            recall = {k: 0.0 for k in [1, 5, 10]}
            for i in range(num_samples):
                for k in recall.keys():
                    if i in indices[:, :k].flatten():
                        recall[k] += 1
            return {k: v / num_samples for k, v in recall.items()}

        i2t_recall = calculate_recall(indices, len(image_features))
        t2i_recall = calculate_recall(indices.T, len(text_features))

        return i2t_recall, t2i_recall

    def eval(self):
        data_loader = self.create_coco_dataloader()

        print("Extracting features...")
        image_features, text_features, all_captions = self.extract_features(data_loader)

        print("Computing metrics...")
        i2t_recall, t2i_recall = self.compute_metrics(image_features, text_features)

        print("Image to Text Recall:")
        for k, v in i2t_recall.items():
            print(f"R@{k}: {v:.4f}")

        print("\nText to Image Recall:")
        for k, v in t2i_recall.items():
            print(f"R@{k}: {v:.4f}")

# 使用示例
# evaluator = Evaluator(tar_dir='path_to_tar_files', batch_size=32, coco_root='path_to_coco_root', ann_file='path_to_ann_file', device='cuda')
# evaluator.eval()