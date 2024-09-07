### torchvision 选择预测正确的图像并保存
```bash
CUDA_VISIBLE_DEVICES=0 python select_from_imagenet.py  \
    --imagenet_path '/mnt/sdc1/data/ILSVRC2012/val' \
    --output_path "/OUTPUT/PATH" \
    --model "resnet50" #['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152','mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small']
```