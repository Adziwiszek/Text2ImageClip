import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, embed_dim=128, img_channels=3):
        super().__init__()
        input_dim = z_dim # + embed_dim  # Concatenate noise and text embedding

        self.net = nn.Sequential(
            nn.Linear(input_dim, 4*4*512),
            nn.BatchNorm1d(4*4*512),
            nn.ReLU(True),
            View((-1, 512, 4, 4)),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),  # 64x64
            nn.Tanh()
        )

    def forward(self, z, text_embed=None):
        # text_embed = text_embed.squeeze(1)
        # x = torch.cat([z, text_embed], dim=1)
        # return self.net(x)
        return (self.net(z) + 1) / 2


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
