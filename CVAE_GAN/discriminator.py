import torch.nn as nn

# from torchtyping import TensorType
# from typeguard import typechecked


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(  # B, 3, 64, 64
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            # B, 32, 32, 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 128, kernel_size=5, stride=2, padding=2),
            # B, 128, 16, 16
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            # B, 256, 8, 8
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # B, 256 * 8 * 8
            nn.Linear(256 * 8 * 8, 512),
            #nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out, features


'''
asf = Discriminator()
test_batch = torch.ones((64, 3, 64, 64))  # [B, channels, xsize, ysize]
test_res = asf(test_batch)
'''
