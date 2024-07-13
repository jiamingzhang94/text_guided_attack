import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.skip_conv(residual)
        out += residual
        return self.relu(out)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.relu(self.bn(self.conv(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim=1024, img_channels=3, img_size=224, eps=None):
        super(Decoder, self).__init__()
        self.embedding_dim = embed_dim
        self.eps = eps
        self.img_channels = img_channels
        self.img_size = img_size
        self.init_size = img_size // 16  # Initial size before upsampling

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 256 * self.init_size ** 2)
        )

        self.upsample_blocks = nn.ModuleList([
            ResBlock(256, 256),
            UpBlock(256, 128),
            ResBlock(128, 128),
            UpBlock(128, 64),
            ResBlock(64, 64),
            UpBlock(64, 32),
            ResBlock(32, 32),
            UpBlock(32, 16),
            ResBlock(16, 16)
        ])

        self.final_conv = nn.Conv2d(16, img_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, embedding):
        embedding = embedding.view(embedding.size(0), -1)
        # embedding = (embedding - 0.5)/0.5
        out = self.fc(embedding.float())
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        for block in self.upsample_blocks:
            out = block(out)
        img = self.final_conv(out)
        img = self.tanh(img)
        img = img * 0.5 + 0.5
        if self.eps is not None:
            img = img * self.eps
        return img


# Example usage
if __name__ == "__main__":
    model = Text2ImgGenerator()
    embedding = torch.randn((8, 1024))  # Batch of 8, each with 1024-dim embedding
    generated_images = model(embedding)
    print(generated_images.shape)  # Expected output: (8, 3, 224, 224)
