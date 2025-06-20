import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, embed_dim=128, img_channels=3):
        super().__init__()

        # Project text embedding and tile spatially
        self.embed_proj = nn.Linear(embed_dim, 64 * 64)

        self.net = nn.Sequential(
            #nn.Conv2d(img_channels + 1, 64, 4, 2, 1),  # 32x32
            nn.Conv2d(img_channels, 64, 4, 2, 1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),  # 16x16
            nn.LayerNorm([128, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),  # 8x8
            nn.LayerNorm([256, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),  # 4x4
            nn.LayerNorm([512, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4),  # Output 1x1
        )

    def forward(self, img, text_embed=None):
        # batch_size = img.size(0)
        # cond = self.embed_proj(text_embed).view(batch_size, 1, 64, 64)
        # x = torch.cat([img, cond], dim=1)  # Concatenate on channel axis
        # out = self.net(x)
        out = self.net(img)
        return out.view(-1)  # Output: scalar per sample
