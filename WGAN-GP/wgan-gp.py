import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import os
import wandb
import clip
from tqdm import tqdm
import torch.multiprocessing as mp
import typer

from my_secrets import wandb_key, wandb_proj_name, celeba_img_path, celeba_attr_path
from generator_model import Generator
from critic_model import Critic
from data_prep import CelebADataset
from common import device


prompts = [
    "A bald woman with glasses",
    "A young attractive woman with bangs, wearing glasses",
    "Old black bald man with beard",
]

def gradient_penalty(critic, real_imgs, fake_imgs, text_embed=0):
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real_imgs + ((1 - alpha) * fake_imgs)
    interpolates.requires_grad_(True)

    # critic_scores = critic(interpolates, text_embed)
    critic_scores = critic(interpolates)

    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty

def generate_and_log_images(generator, clip_model):
    generator.eval()
    images = []
    for prompt in prompts:
        with torch.no_grad():
            text = clip.tokenize([prompt]).to(device)
            text_embed = clip_model.encode_text(text).float()
            z = torch.randn(1, 100, device=device)
            fake_image = generator(z, text_embed).detach().cpu()
            images.append(wandb.Image(fake_image[0], caption=prompt))
    wandb.log({"Generated Images": images})

def train(generator, critic, dataloader, g_opt, c_opt, clip_model, num_epochs):
    lambda_gp = 10
    critic_iters = 5

    for epoch in range(1, num_epochs + 1):
        for i, (real_imgs, _, clip_emb) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            real_imgs = real_imgs.to(device)
            # clip_emb = clip_emb.to(device)
            batch_size = real_imgs.size(0)

            # === Train Critic ===
            for _ in range(critic_iters):
                z = torch.randn(batch_size, 100, device=device)
                # fake_imgs = generator(z, clip_emb).detach()
                # c_real = critic(real_imgs, clip_emb).mean()
                # c_fake = critic(fake_imgs, clip_emb).mean()
                # gp = gradient_penalty(critic, real_imgs, fake_imgs, clip_emb)
                fake_imgs = generator(z).detach()
                c_real = critic(real_imgs).mean()
                c_fake = critic(fake_imgs).mean()
                gp = gradient_penalty(critic, real_imgs, fake_imgs)
                c_loss = c_fake - c_real + lambda_gp * gp
                c_opt.zero_grad()
                c_loss.backward()
                c_opt.step()

            # === Train Generator ===
            z = torch.randn(batch_size, 100, device=device)
            fake_imgs = generator(z, clip_emb)
            g_loss = -critic(fake_imgs, clip_emb).mean()
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

        wandb.log({"Generator Loss": g_loss.item(), "Critic Loss": c_loss.item()})
        print(f"Epoch {epoch}: G Loss: {g_loss.item():.4f}, C Loss: {c_loss.item():.4f}")
        generate_and_log_images(generator, clip_model)

def run_training():
    wandb.login(key=wandb_key)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    for p in clip_model.parameters():
        p.requires_grad = False

    df_attrs = pd.read_csv(celeba_attr_path,
        skiprows=1,
        index_col=0,
        delim_whitespace=True
    )
    attributes = [col for col in df_attrs.columns]

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = CelebADataset(
        img_dir=celeba_img_path,
        attr_df=df_attrs,
        transform=transform,
        attributes=attributes,
        clip_model=clip_model,
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    generator = Generator(z_dim=100, embed_dim=512).to(device)
    critic = Critic(embed_dim=512).to(device)

    g_opt = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    c_opt = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))

    wandb.init(project=wandb_proj_name, config={"model": "WGAN-GP", "epochs": 100})
    train(generator, critic, dataloader, g_opt, c_opt, clip_model, num_epochs=100)

    torch.save({
        'generator_state_dict': generator.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'g_opt_state_dict': g_opt.state_dict(),
        'c_opt_state_dict': c_opt.state_dict(),
    }, 'checkpoint.pth')
    wandb.save("generator.pth")
    wandb.finish()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_training()