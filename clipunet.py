import os
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from mmpretrain import FeatureExtractor,get_model
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel,CLIPVisionModel
import torchvision.transforms as transforms
class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv3d = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv3d(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, inChannel, outChannel, fChannel=32, bilinear=True):
        super(UNet3D, self).__init__()
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.fChannel=fChannel
        self.bilinear = bilinear

        self.inc = DoubleConv3D(inChannel, fChannel)
        self.down1 = Down3D(fChannel, fChannel*2)
        self.down2 = Down3D(fChannel*2, fChannel*4)
        self.down3 = Down3D(fChannel*4, fChannel*8)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(fChannel*8, fChannel*16 // factor)

        self.up1 = Up3D(fChannel*16, fChannel*8 // factor, bilinear)
        self.up2 = Up3D(fChannel*8, fChannel*4 // factor, bilinear)
        self.up3 = Up3D(fChannel*4, fChannel*2 // factor, bilinear)
        self.up4 = Up3D(fChannel*2, fChannel, bilinear)
        self.outc = OutConv3D(fChannel, outChannel)

    def forward(self, x,args=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    def forward_x1(self,x):
        x1 = self.inc(x)
        return x1

class UNet3D_PSA(nn.Module):
    def __init__(self, inChannel, outChannel, fChannel=32, bilinear=True):
        super(UNet3D_PSA, self).__init__()
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.fChannel=fChannel
        self.bilinear = bilinear

        self.inc = DoubleConv3D(inChannel, fChannel)
        self.down1 = Down3D(fChannel, fChannel*2)
        self.down2 = Down3D(fChannel*2, fChannel*4)
        self.down3 = Down3D(fChannel*4, fChannel*8)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(fChannel*8, fChannel*16 // factor)

        #decoder
        self.up1 = nn.ModuleList([Up3D(fChannel*16, fChannel*8 // factor, bilinear) for i in range(5)])
        self.up2 = nn.ModuleList([Up3D(fChannel*8, fChannel*4 // factor, bilinear)for i in range(5)])
        self.up3 = nn.ModuleList([Up3D(fChannel*4, fChannel*2 // factor, bilinear)for i in range(5)])
        self.up4 = nn.ModuleList([Up3D(fChannel*2, fChannel, bilinear)for i in range(5)])
        self.outc = nn.ModuleList([OutConv3D(fChannel, outChannel)for i in range(5)])
        self.train_decoder0=True

    def forward(self, x, sentence):
        if self.train_decoder0:
            decoder=0
            self.train_decoder0=False
        else:
            if sentence[-2]=='A':#Desai, Neil
                decoder=1
            elif sentence[-2]=='D':#Hannan, Raquibul
                decoder=2
            elif sentence[-2]=='I':#Yang, Daniel
                decoder=3
            elif sentence[-2]=='J':#Garant, Aurelie
                decoder=4
            else:
                decoder=0 #Others
            self.train_decoder0=True

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1[decoder](x5, x4)
        x = self.up2[decoder](x, x3)
        x = self.up3[decoder](x, x2)
        x = self.up4[decoder](x, x1)
        logits = self.outc[decoder](x)
        return logits

class CLIPUNet3D(nn.Module):
    def __init__(self, inChannel, outChannel, fChannel=32, bilinear=True):
        super(CLIPUNet3D, self).__init__()
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.fChannel=fChannel
        self.bilinear = bilinear

        self.inc = DoubleConv3D(inChannel, fChannel)
        self.down1 = Down3D(fChannel, fChannel*2)
        self.down2 = Down3D(fChannel*2, fChannel*4)
        self.down3 = Down3D(fChannel*4, fChannel*8)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(fChannel*8, fChannel*16 // factor)

        self.up1 = Up3D(fChannel*16, fChannel*8 // factor, bilinear)
        self.up2 = Up3D(fChannel*8, fChannel*4 // factor, bilinear)
        self.up3 = Up3D(fChannel*4, fChannel*2 // factor, bilinear)
        self.up4 = Up3D(fChannel*2, fChannel, bilinear)
        self.outc = OutConv3D(fChannel, outChannel)

        self.clip_model, _ = clip.load("ViT-B/32", device='cpu')
        self.downtext=nn.AvgPool1d(kernel_size=2,stride=2)
        #text = clip.tokenize([r'There is no spacer hydrogel in the patient.',r'There is a spacer hydrogel in the patient.'])
        #text = clip.tokenize([r'There is a type 2 spacer hydrogel in the patient.',r'There is no type 2 spacer hydrogel in the patient.'])
        return

    def forward(self, x, text):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        text=clip.tokenize(text).to(x.device)
        text_feature = self.clip_model.encode_text(text)
        text_feature.unsqueeze_(dim=1).detach_()
        text_feature=self.downtext(text_feature)
        text_feature=(text_feature-0.015)/0.27
        #text_feature=self.downtext(text_feature)
        x5=x5*text_feature.view(1,text_feature.shape[2],1,1,1)
        x4=x4*text_feature.view(1,text_feature.shape[2],1,1,1)
        text_feature=self.downtext(text_feature)
        x3=x3*text_feature.view(1,text_feature.shape[2],1,1,1)
        text_feature=self.downtext(text_feature)
        x2=x2*text_feature.view(1,text_feature.shape[2],1,1,1)
        text_feature=self.downtext(text_feature)
        x1=x1*text_feature.view(1,text_feature.shape[2],1,1,1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    # model1=UNet3D_PSA(1,1,fChannel=32)
    # input=torch.randn(size=[1,1,64,64,64])
    # output1=model1(input,0)
    # pass

    # /data2/ModelWarehouse/clip-vit-base-patch32
    # model = CLIPVisionModel.from_pretrained("/data2/ModelWarehouse/clip-vit-base-patch32")
    # processor = CLIPProcessor.from_pretrained("/data2/ModelWarehouse/clip-vit-base-patch32")
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # # inputs = processor(images=image, return_tensors="pt")
    # # outputs = model(**inputs,output_hidden_states=True,output_attentions=True)
    # # print("1")
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    # tensor_image = transform(image)
    # # 添加批次维度
    # tensor_image = tensor_image.unsqueeze(0)
    #
    # # 复制张量5次以形成新的形状 (5, 3, 224, 224)
    # tensor_image_replicated = tensor_image.repeat(5, 1, 1, 1)

    model=CLIPUNet3D(3,3)
    input=torch.randn(size=[5,3,64,64,64])

    output=model(input,)
    print("1")