import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, image_size):
        super(Critic, self).__init__()

        self.image_size = image_size
        self.disc = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._block(64,  128, 4, 2, 1),
            self._block(128, 256, 4, 2, 1),
            self._block(256, 512, 4, 2, 1),
            nn.Conv2d(512, 1, 4, 2, 0)
        )
        self.embedding = nn.Embedding(10, image_size*image_size)

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

    def forward(self, x, labels):
        embedding = self.embedding(labels).view(labels.shape[0], 1, self.image_size, self.image_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, image_size, emb_size):
        super(Generator, self).__init__()

        self.image_size = image_size
        self.emb_size = emb_size
        self.gen = nn.Sequential(
            self._block(z_dim + emb_size, 1024, 4, 1, 0),
            self._block(1024, 512, 4, 2, 1),
            self._block(512, 256, 4, 2, 1),
            self._block(256, 128, 4, 2, 1),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )
        self.embedding = nn.Embedding(10, emb_size)
        
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

    def forward(self, z, labels):
        embedding = self.embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([z, embedding], dim=1)
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, 
                          nn.ConvTranspose2d, 
                          nn.BatchNorm2d, 
                          nn.InstanceNorm2d, 
                          nn.Embedding)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
