import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# --- Generator definition (same as yours) ---
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
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, z, labels):
        embedding = self.embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([z, embedding], dim=1)
        return self.gen(x)

# --- Load Generator ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100
emb_size = 100
image_size = 64

checkpoint = torch.load("checkpoint2.pth", map_location=device)
G = Generator(z_dim=z_dim, image_size=image_size, emb_size=emb_size).to(device)
G.load_state_dict(checkpoint["generator_state_dict"])
G.eval()

# --- Generate one sample per digit ---
def generate_digit_images():
    z = torch.randn(10, z_dim, 1, 1).to(device)
    labels = torch.arange(10).to(device)
    with torch.no_grad():
        fakes = G(z, labels).cpu()
    return fakes, labels.cpu()

generated_images, gen_labels = generate_digit_images()

# --- Load MNIST test set, resize to 64x64 and normalize to [-1, 1] ---
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = MNIST(root="dataset/", train=False, download=True, transform=transform)
mnist_loader = DataLoader(mnist, batch_size=256, shuffle=False)

# --- Collect all real images in a single tensor ---
real_images = []
real_labels = []

for imgs, labels in tqdm(mnist_loader, desc="Loading MNIST"):
    real_images.append(imgs)
    real_labels.append(labels)

real_images = torch.cat(real_images, dim=0)  # shape: [10000, 1, 64, 64]
real_labels = torch.cat(real_labels, dim=0)

# --- Find 5 closest MNIST digits to each generated digit ---
def find_closest(fake, real_set, top_k=5):
    fake_flat = fake.view(1, -1)
    real_flat = real_set.view(real_set.size(0), -1)
    distances = torch.norm(real_flat - fake_flat, dim=1)
    indices = torch.topk(distances, k=top_k, largest=False).indices
    return real_set[indices]

# --- Plot results ---
fig, axes = plt.subplots(10, 6, figsize=(10, 12))
for i in range(10):
    fake_img = generated_images[i]
    closest_imgs = find_closest(fake_img, real_images[real_labels == i], top_k=5)

    # Generated digit
    axes[i][0].imshow(fake_img.squeeze().numpy(), cmap="gray")
    axes[i][0].set_title(f"Gen {i}")
    axes[i][0].axis("off")

    # 5 closest real digits
    for j in range(5):
        axes[i][j+1].imshow(closest_imgs[j].squeeze().numpy(), cmap="gray")
        axes[i][j+1].set_title(f"Real")
        axes[i][j+1].axis("off")

plt.tight_layout()
plt.show()