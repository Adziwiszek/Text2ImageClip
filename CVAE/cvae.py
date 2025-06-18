import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
from PIL import Image
import os
import wandb
import clip
from tqdm import tqdm
import multiprocessing as mp
import typer

from my_secrets import wandb_key, wandb_proj_name, celeba_img_path, celeba_attr_path
from .model import ClipCVAE
from .data_prep import CelebADataset
from .common import device


# Prompts for generating images each epoch
prompts = [
    "A bald woman with glasses",
    "A young attractive woman with bangs, wearing glasses",
    "Old black bald man with beard",
]


def generate_and_log_images(model):
    images = [wandb.Image(model.generate_image(prompt), caption=prompt)
              for prompt in prompts]
    wandb.log({"Generated image": images})


def loss_function(recon_x, x, mu, logvar, criterion):
    recon_loss = criterion(recon_x, x)
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KL


def train(model, dataloader, optimizer, criterion, num_epochs):
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        for _, (data, attrs, clip_emb) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
            data = data.to(device, non_blocking=True)
            attrs = attrs.to(device, non_blocking=True)
            clip_emb = clip_emb.to(device, non_blocking=True)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, attrs, clip_emb)
            loss = loss_function(recon_batch, data, mu, logvar, criterion)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(dataloader.dataset)
        wandb.log({"train_loss": avg_loss})
        print(f'Epoch: {epoch}, Avg loss:{avg_loss:.4f}')

        generate_and_log_images(model)

    return model


def run_training():
    wandb.login(key=wandb_key)

    # Load CLIP model from openai
    clip_model, _ = clip.load("ViT-B/32", device=device)
    for param in clip_model.parameters():
        param.requires_grad = False

    # Loading image attributes
    df_attrs = pd.read_csv(celeba_attr_path)

    attributes = [col for col in df_attrs.columns]
    # attribute2id = {att: id for id, att in enumerate(attributes)}
    # id2attribute = {id: att for id, att in enumerate(attributes)}

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

    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4)

    latent_dim = 128
    cond_dim = 40
    clip_dim = 512
    model = ClipCVAE(
        clip_model,
        img_channels=3,
        img_size=64,
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        clip_dim=clip_dim) \
        .to(device)

    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')

    # model training
    num_epochs = 5

    wandb.init(
        project=wandb_proj_name,
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "fresh_start": False,
            "batch_norm": False,
            }
    )

    train(model, dataloader, optimizer, criterion, num_epochs)
    # model.load_state_dict(torch.load('model.pth'))

    # Saving model parameters
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact("model_params", type="model")
    artifact.add_file("model.pth")
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    run_training()
