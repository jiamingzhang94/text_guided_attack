from tqdm import tqdm
import json
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import os
import argparse
# os.environ['TORCH_HOME'] = '/new_data/yifei2/junhong/text_guide_attack/cache'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class imagenet(datasets.ImageFolder):
    def __getitem__(self, index):
        image,label = super().__getitem__(index)
        path, _ = self.samples[index]
        label=int(path.split('/')[-2])
        # label=self.class_to_idx[label]
        return image, label,path

def get_model(model_name, device):
    """
    根据模型名称返回对应的预训练模型及其权重。
    """
    # 映射模型名称到 torchvision 的模型函数和权重
    model_mapping = {
        'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
        'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT),
        'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
        'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT),
        'resnet152': (models.resnet152, models.ResNet152_Weights.DEFAULT),
        # 可以根据需要添加更多模型
        'mobilenet_v2': (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
        'mobilenet_v3_large': (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
        'mobilenet_v3_small': (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
        # 例如： 'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    }

    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models are: {list(model_mapping.keys())}")

    model_func, weights = model_mapping[model_name]
    model = model_func(weights=weights)
    model = model.to(device)
    model.eval()
    return model, weights

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet_path', type=str, default='/mnt/sdc1/data/ILSVRC2012/val')
    parser.add_argument('--save_path', type=str, default='imagenet_resnet101.json')
    parser.add_argument('--model', type=str, default='resnet101',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small'],
                        help='选择要使用的模型架构')
    args = parser.parse_args()

    model, weights = get_model(args.model, device)

    # 设置数据转换
    transform = weights.transforms()

    # 加载 ImageNet 测试集
    imagenet_data = imagenet(root=args.imagenet_path, transform=transform)
    data_loader = DataLoader(imagenet_data, batch_size=1, shuffle=False)

    # 创建一个字典来存储每个类别的正确分类图像的文件名和标签
    correct_images = {}

    # 用于跟踪已经找到的类别
    found_classes = set()

    num=0
    # 遍历测试集
    with torch.no_grad():
        for i, (inputs, labels,image_path) in enumerate(tqdm(data_loader)):
            # print(labels)
            # print(inputs.shape)
            # 如果该类已找到正确预测的图像，则跳过
            if labels.item() in found_classes:
                continue
            # 将输入和标签转移到 GPU 上
            inputs = inputs.to(device)
            labels = labels.to(device)

            prediction = model(inputs).squeeze(0).softmax(0)

            # Make predictions
            preds = prediction.argmax().item()

            # 检查预测是否正确
            if preds == labels.item():
                relative_path = '/'.join(image_path[0].split('/')[-3:])
                correct_images[labels.item()] = {
                    'file_name': relative_path,
                    'label': labels.item(),
                    'predicted_label': preds
                }
                found_classes.add(labels.item())  # 记录已经找到的类别
                num+=1
                print(num,labels.item())

            # 如果所有类别都已找到，则停止遍历
            if len(correct_images) == len(imagenet_data.classes):
                break
            break
    # 将结果保存为 JSON 文件
    output_file = args.save_path
    with open(output_file, "w",encoding='utf-8') as f:
        json.dump(correct_images, f, indent=4,ensure_ascii=False)

    print(f"Results saved to {output_file}")
    # # 输出结果
    for class_id, info in correct_images.items():
        file_name = info['file_name']
        label = info['label']
        predicted_label = info['predicted_label']
        print(f"File Name: {file_name}, Label: {label}")
