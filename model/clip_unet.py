import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
# from mmpretrain import FeatureExtractor, get_model
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import argparse


class CLIP_Vision_encoder(nn.Module):
    def __init__(self, args):
        super(CLIP_Vision_encoder, self).__init__()
        self.args = args
        self.encoder = CLIPVisionModel.from_pretrained(args.clip_model_path)
        self.processer = CLIPProcessor.from_pretrained(self.args.clip_model_path)

    def forward(self, x):
        inputs = self.processer(images=x, return_tensors="pt")
        encode = self.encoder(**inputs, output_hidden_states=True, output_attentions=True)
        last_hidden_state = encode.last_hidden_state
        return last_hidden_state


class CLIP_Text_encoder(nn.Module):
    def __init__(self, args):
        super(CLIP_Text_encoder, self).__init__()
        self.args = args
        self.encoder = CLIPTextModel.from_pretrained(self.args.clip_model_path)
        self.Tokenizer = CLIPTokenizer.from_pretrained(self.args.clip_model_path)

    def forward(self, x):
        input = self.Tokenizer(x, padding=True, return_tensors="pt")
        encode = self.encoder(input_ids=input["input_ids"], output_hidden_states=True, output_attentions=True)
        last_hidden_state = encode.last_hidden_state
        return last_hidden_state


class Deconv(nn.Module):
    def __init__(self, input_chanel, output_chanel):
        super(Deconv, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_chanel, output_chanel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class CLIP_encoder_decoder(nn.Module):
    def __init__(self, args):
        super(CLIP_encoder_decoder, self).__init__()
        self.args = args
        # self.train=args.train
        # CLIPMoedel : emb_shape [batch_size,1,512]
        self.processor = CLIPProcessor.from_pretrained(self.args.clip_model_path)
        self.encoder = CLIPModel.from_pretrained(args.clip_model_path)
        for param in self.encoder.parameters():
            param.requires_grad = False
        # clip 编码得到的特征(batch_size,512)->(batch_size,512,49)
        self.linear = nn.Linear(1, 49)

        self.deconv5 = Deconv(512, 256)
        self.deconv4 = Deconv(256, 128)
        self.deconv3 = Deconv(128, 64)
        self.deconv2 = Deconv(64, 32)
        self.deconv1 = Deconv(32, 3)

        # self.tmp_image = Image.open("/data2/zhiyu/data/coco/images/val2014/COCO_val2014_000000000042.jpg").convert(
        #     'RGB')

        # self.device = args.device

    def forward(self, inputs, train=True):
        with torch.no_grad():
            if train:
                # inputs=self.processor(text=[""],images=x, return_tensors="pt", padding=True).cuda()
                feature = self.encoder(**inputs)  # [B,1,512]->[B,49,512] 1*1卷积
                feature = feature.image_embeds
            else:
                # inputs = self.processor(text=x,images=[self.tmp_image],return_tensors="pt", padding=True).cuda()
                feature = self.encoder(**inputs)
                feature = feature.text_embeds
        #  (batch_size, 1,512) 转换为 (batch_size * 512, 1)
        reshaped_input = feature.unsqueeze(1).permute(0, 2, 1).contiguous().view(-1, 1)
        center_feature = self.linear(reshaped_input)

        center_feature = center_feature.view(-1, 512, 7, 7)

        deconv_feature5 = self.deconv5(center_feature)
        deconv_feature4 = self.deconv4(deconv_feature5)
        deconv_feature3 = self.deconv3(deconv_feature4)
        deconv_feature2 = self.deconv2(deconv_feature3)
        image = self.deconv1(deconv_feature2)
        image=torch.clamp(image, self.args.epsilon / 255., self.args.epsilon / 255)
        # inputs=self.processor(text=[""],images=x, return_tensors="pt", padding=True)
        output_encode = self.encoder(pixel_values=image,
                                     input_ids=torch.tensor([[49406, 49407]]).cuda())  # [B,1,512]->[B,49,512] 1*1卷积
        output_encode_feature = output_encode.image_embeds

        return image, feature, output_encode_feature


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--train", type=str, default=True)
    argparse.add_argument("--clip_model_path", type=str, default="/data2/ModelWarehouse/clip-vit-base-patch32")
    args = argparse.parse_args()
    # encoder = CLIP_Vision_encoder(args=args)
    # model = CLIP_encoder_decoder(args=args)
    # # 数据预处理和加载
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    url = "/data2/zhiyu/data/coco/images/val2014/COCO_val2014_000000000042.jpg"
    image = Image.open(url).convert('RGB')
    model = CLIP_encoder_decoder(args=args)
    model=model.to(device)

    processor = CLIPProcessor.from_pretrained(args.clip_model_path)
    input=processor(text=["hello", "hello"],images=image, return_tensors="pt", padding=True)
    input=input.to(device)
    # output, feature, output_encode_feature = model([image, image])
    output, feature,output_encode_feature = model(input,train=False)
    print(output.shape)
    print(feature.shape)
    print(output_encode_feature.shape)
    # image = transform(image)
    # image = torch.randn(size=[5,3,224,224])
    # image="hello,hello,hello"
    # print(model([image,image,image]).shape)
    # train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # model = CLIPModel.from_pretrained(args.clip_model_path)
    # processor = CLIPProcessor.from_pretrained(args.clip_model_path)

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    # inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=[image,image], return_tensors="pt", padding=True)
    #
    # outputs = model(**inputs)
    # print("1")
    # print()
