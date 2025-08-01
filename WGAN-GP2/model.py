import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, attr_dim, image_size):
        super(Critic, self).__init__()

        self.attr_dim = attr_dim
        self.image_size = image_size
        self.disc = nn.Sequential(
            nn.Conv2d(3 + attr_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._block(64,  128, 4, 2, 1),
            self._block(128, 256, 4, 2, 1),
            self._block(256, 512, 4, 2, 1),
            nn.Conv2d(512, 1, 4, 2, 0)
        )
        self.embedding = nn.Embedding(attr_dim * 2, image_size*image_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, attr):
        embedding = self.embedding(attr).view(attr.shape[0], self.attr_dim, self.image_size, self.image_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, attr_dim, image_size):
        super(Generator, self).__init__()

        self.image_size = image_size
        self.gen = nn.Sequential(
            self._block(z_dim + attr_dim, 1024, 4, 1, 0),
            self._block(1024, 512, 4, 2, 1),
            self._block(512, 256, 4, 2, 1),
            self._block(256, 128, 4, 2, 1),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, z, attr):
        attr = attr.unsqueeze(2).unsqueeze(3)
        # print(z.size(), attr.size())
        x = torch.cat([z, attr], dim=1)
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, 
                          nn.ConvTranspose2d, 
                          nn.BatchNorm2d, 
                          nn.InstanceNorm2d, 
                          nn.Embedding)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
