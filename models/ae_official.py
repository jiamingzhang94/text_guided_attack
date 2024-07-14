import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import torchvision.transforms as transforms
import models.clip as clip


class CLIPEncoder(nn.Module):
    def __init__(self, model="ViT-B/32"):
        """
        CLIP Image Encoder using the official CLIP implementation.

        Args:
            model (str): The model name. Supported models are:
                "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px",
                "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64".
        """
        super(CLIPEncoder, self).__init__()
        self.model, _ = clip.load(model)
        self.model.eval()  # Set the model to evaluation mode

    def encode_img(self, images):
        """
        Forward pass for the CLIP image encoder.

        Args:
            images (torch.Tensor): A batch of images with shape (batch_size, 3, height, width).
                                   The images should be in the range [0, 1].

        Returns:
            torch.Tensor: Image embeddings with shape (batch_size, 512).
        """
        assert images.ndim == 4 and images.shape[
            1] == 3, "Input images should have shape (batch_size, 3, height, width)"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images = normalize(images)

        image_features = self.model.encode_image(images)
        return image_features

    def encode_text(self, texts):
        """
        Forward pass for the CLIP text encoder.

        Args:
            texts (list): A list of strings to encode.

        Returns:
            torch.Tensor: Text embeddings with shape (batch_size, 512).
        """
        text_tokens = clip.tokenize(texts).to(next(self.model.parameters()).device)
        text_features = self.model.encode_text(text_tokens)
        return text_features


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv1_1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv2_1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1_1x1(self.conv1(x))))
        x = self.bn2(self.conv2_1x1(self.conv2(x)))
        x += residual
        return F.relu(x)


class EfficientAttention(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super(EfficientAttention, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = F.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention

class Decoder(nn.Module):
    def __init__(self, embed_dim=512, latent_dim=64, init_size=7, last_channel=3, num_res_blocks=1):
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.init_size = init_size

        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, latent_dim * self.init_size * self.init_size),
            nn.ReLU(inplace=True)
        )

        self.init_conv = nn.Conv2d(latent_dim, 32, kernel_size=3, padding=1)

        self.up_blocks = nn.ModuleList([
            self._make_up_block(32, 32),
            self._make_up_block(32, 16),
            self._make_up_block(16, 16),
            self._make_up_block(16, 8),
            self._make_up_block(8, last_channel),
            # self._make_up_block(16, 16),
        ])

        self.res_blocks = nn.ModuleList([
            self._make_res_block(last_channel) for _ in range(num_res_blocks)
        ])

        self.efficient_attention = EfficientAttention(
            in_channels=last_channel,  # 或者你期望的通道数
            key_channels=32,  # 可以调整
            head_count=4,  # 可以调整
            value_channels=32  # 可以调整
        )

        self.final_conv = nn.Conv2d(last_channel, 3, kernel_size=3, padding=1)

    def _make_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_res_block(self, channels):
        return ResidualBlock(channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        assert x.size(1) == self.embed_dim
        x = self.embed_proj(x.float())
        x = x.view(-1, self.latent_dim, self.init_size, self.init_size)

        x = self.init_conv(x)

        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.efficient_attention(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.final_conv(x)
        x = torch.tanh(x)
        x = x*0.5 + 0.5

        return x



