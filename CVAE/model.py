import torch
import torch.nn as nn
from PIL import Image

from .common import device, generate_text_embeddings


class ClipCVAE(nn.Module):
    def __init__(
            self,
            clip_model,
            img_channels=3,
            img_size=64,
            latent_dim=128,
            cond_dim=40,
            clip_dim=512
    ):
        super(ClipCVAE, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.clip_dim = clip_dim
        self.clip_model = clip_model

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels + cond_dim + clip_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(512*4*4, latent_dim)
        self.fc_logvar = nn.Linear(512*4*4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + cond_dim + clip_dim, 512*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x, c, clip_embedding):
        c = c.view(c.size(0), self.cond_dim, 1, 1) \
            .repeat(1, 1, self.img_size, self.img_size)
        clip_embedding = \
            clip_embedding.view(clip_embedding.size(0), self.clip_dim, 1, 1) \
            .repeat(1, 1, self.img_size, self.img_size)
        x = torch.cat([x, c, clip_embedding], dim=1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c, clip_embedding):
        clip_embedding = clip_embedding.squeeze(1)
        z = torch.cat([z, c, clip_embedding], dim=1)
        x = self.decoder_input(z)
        x = x.view(-1, 512, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x, c, clip_embedding):
        mu, logvar = self.encode(x, c, clip_embedding)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c, clip_embedding)
        return recon_x, mu, logvar

    def generate_image(self, prompt, c=None):
        with torch.no_grad():
            self.eval()
            text_embedding = generate_text_embeddings(prompt, self.clip_model).to(device)

            z = torch.randn(1, self.latent_dim).to(device)

            if c is None:
                c = torch.zeros(1, self.cond_dim).to(device)

            recon = self.decode(z, c, text_embedding)

            recon = recon.squeeze(0).detach().cpu()
            recon = recon.clamp(0, 1)
            recon_img = (recon * 255).byte().permute(1, 2, 0).numpy()

            return Image.fromarray(recon_img)
