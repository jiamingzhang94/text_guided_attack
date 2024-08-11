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

def get_group_norm(num_channels, num_groups=32):
    if num_channels < num_groups:
        return nn.GroupNorm(num_groups=num_channels, num_channels=num_channels)
    return nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = get_group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = get_group_norm(channels)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.gn1(self.conv1(x)), negative_slope=0.2)
        out = self.gn2(self.conv2(out))
        out += residual
        return F.leaky_relu(out, negative_slope=0.2)

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
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention

class Decoder(nn.Module):
    def __init__(self, embed_dim=512, latent_dim=512, init_size=7, last_channel=16, num_res_blocks=3):
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.init_size = init_size

        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, latent_dim * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.init_conv = nn.Conv2d(latent_dim, 256, kernel_size=3, padding=1)

        self.up_blocks = nn.ModuleList([
            self._make_up_block(256, 128),
            self._make_up_block(128, 64),
            self._make_up_block(64, 32),
            self._make_up_block(32, 16),
            self._make_up_block(16, last_channel),
        ])

        self.skip_projections = nn.ModuleList([
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.Conv2d(16, last_channel, kernel_size=1),
        ])

        self.res_blocks = nn.ModuleList([
            ResidualBlock(last_channel) for _ in range(num_res_blocks)
        ])

        self.efficient_attention = EfficientAttention(
            in_channels=last_channel,
            key_channels=32,
            head_count=4,
            value_channels=32
        )

        self.final_conv = nn.Conv2d(last_channel, 3, kernel_size=3, padding=1)
        self.gn = get_group_norm(last_channel)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            get_group_norm(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(out_channels)
        )

    def forward(self, x):
        x = self.embed_proj(x.view(x.size(0), -1).float())
        x = x.view(-1, self.latent_dim, self.init_size, self.init_size)

        x = self.init_conv(x)

        features = []
        for up_block, skip_proj in zip(self.up_blocks, self.skip_projections):
            features.append(F.interpolate(skip_proj(x), scale_factor=2, mode='nearest'))
            x = up_block(x)
            if features:
                x = x + features[-1]  # 添加跳跃连接

        x = self.gn(x)
        x = self.efficient_attention(x)
        x = self.gn(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.final_conv(x)
        x = torch.tanh(x)
        x = x * 0.5 + 0.5

        return x