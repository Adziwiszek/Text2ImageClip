import torch
import torch.nn as nn
import wandb
import tqdm
import pandas as pd
import torch.optim as optim
import multiprocessing as mp
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import clip
'''
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
'''

from my_secrets import wandb_key, wandb_proj_name, celeba_img_path, celeba_attr_path
from .model import CVAE_GAN
from ..CVAE.data_prep import CelebADataset
from generate_data import generate_and_log_images, generate_from_prompts

device = 'cuda'


def train_vaegan(model: CVAE_GAN,
                 optimizer_encoder,
                 optimizer_decoder,
                 optimizer_discriminator,
                 dataloader,
                 dataset,
                 num_epochs=5):
    gamma = 1
    batches = len(dataloader.dataset)

    for epoch in range(1, num_epochs + 1):
        generate_and_log_images(model, dataset, device)
        generate_from_prompts(model)

        model.train()
        encoder_loss = 0
        decoder_loss = 0
        discriminator_loss = 0
        for _, (x, attrs, clip_emb) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
            x = x.to(device, non_blocking=True)
            attrs = attrs.to(device, non_blocking=True)
            clip_emb = clip_emb.to(device, non_blocking=True)

            # Encode X with reparameterization trick
            mu, logvar = model.encode(x, attrs, clip_emb)
            z = model.reparameterize(mu, logvar)

            # Decode X_tilde with decoder
            x_tilde = model.decode(z, attrs, clip_emb)

            # Sample from prior N(0, I) and decode
            z_p = torch.randn_like(z)
            x_p = model.decode(z_p, attrs, clip_emb)

            # Discriminating
            disc_class_real, disc_feats_real = model.discriminator(x)
            disc_class_pred, disc_feats_pred = model.discriminator(x_tilde)
            disc_class_sampled, disc_feats_sampled = model.discriminator(x_p)

            kl, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled = \
                CVAE_GAN.loss(
                        disc_feats_real, disc_feats_pred, disc_feats_sampled,
                        disc_class_real, disc_class_pred, disc_class_sampled,
                        mu, logvar
                        )

            loss_encoder = torch.sum(kl) + torch.sum(mse)
            loss_discriminator = torch.sum(bce_dis_original) + \
                torch.sum(bce_dis_predicted) + torch.sum(bce_dis_sampled)
            loss_decoder = torch.sum(gamma * mse) - (1.0 - gamma) * \
                loss_discriminator

            model.zero_grad()

            # encoder
            loss_encoder.backward(retain_graph=True)
            optimizer_encoder.step()
            model.zero_grad()

            # decoder
            loss_decoder.backward(retain_graph=True)
            optimizer_decoder.step()
            model.discriminator.zero_grad()

            # discriminator
            loss_discriminator.backward()
            optimizer_discriminator.step()

        wandb.log({
            "encoder_loss": encoder_loss / batches,
            "decoder_loss": decoder_loss / batches,
            "discriminator_loss": discriminator_loss / batches,
            })

    return model


if __name__ == '__main__':
    wandb.login(key=wandb_key)

    # Load CLIP model from openai
    clip_model, _ = clip.load("ViT-B/32", device=device)
    for param in clip_model.parameters():
        param.requires_grad = False

    # Loading image attributes
    df_attrs = pd.read_csv(celeba_attr_path)

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

    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4)

    model = CVAE_GAN(clip_model).to(device)

    lr = 1e-3
    optimizer_encoder = optim.RMSprop(params=model.encoder.parameters(),
                                      lr=lr, alpha=0.9, eps=1e-8,
                                      weight_decay=0, momentum=0,
                                      centered=False)
    optimizer_decoder = optim.RMSprop(params=model.decoder.parameters(),
                                      lr=lr, alpha=0.9, eps=1e-8,
                                      weight_decay=0, momentum=0,
                                      centered=False)
    optimizer_discriminator = optim.RMSprop(
            params=model.discriminator.parameters(),
            lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0,
            centered=False)

    # model training
    num_epochs = 1

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

    train_vaegan(model, optimizer_encoder, optimizer_decoder,
                 optimizer_discriminator, dataloader, dataset, num_epochs)
    # model.load_state_dict(torch.load('model.pth'))

    # Saving model parameters
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact("model_params", type="model")
    artifact.add_file("model.pth")
    wandb.log_artifact(artifact)

    wandb.finish()
