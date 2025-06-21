import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
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
from CVAE.data_prep import CelebADataset
from generate_data import generate_from_prompts, log_reconstructed_images

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
        #log_reconstructed_images(model, dataset, device)
        #generate_from_prompts(model)

        encoder_loss = 0
        decoder_loss = 0
        discriminator_loss = 0
        for i, (x, attrs, clip_emb) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
            model.train()
            print(f'batch {i}')
            x = x.to(device, non_blocking=True)
            attrs = attrs.to(device, non_blocking=True)
            clip_emb = clip_emb.to(device, non_blocking=True)

            with torch.autograd.set_detect_anomaly(True):
                # Forward pass ----------------------------------------

                # Encode X with reparameterization trick
                mu, logvar = model.encode(x, attrs, clip_emb)
                z = model.reparameterize(mu, logvar)

                # Reconstruct and sample from prior
                x_tilde = model.decode(z, attrs, clip_emb)
                z_p = torch.randn_like(z, requires_grad=True)
                x_p = model.decode(z_p, attrs, clip_emb)

                # Discriminating
                concat = torch.cat([x, x_tilde, x_p], dim=0)
                class_logits, feat_maps = model.discriminator(concat)

                B = x.size(0)
                real_logits   = class_logits[:B]
                recon_logits  = class_logits[B:2*B]
                sampled_logits= class_logits[2*B:]

                real_feats    = feat_maps[:B]
                recon_feats   = feat_maps[B:2*B]
                sampled_feats = feat_maps[2*B:]

                # Calculate losses ----------------------------------------
                # KL divergence
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

                # Feature‑matching MSE (for encoder & decoder)
                mse_enc = 0.5 * torch.sum((real_feats.detach() - recon_feats).pow(2), dim=(1,2,3))
                mse_dec = 0.5 * torch.sum((real_feats   - recon_feats).pow(2), dim=(1,2,3))

                # BCE losses
                eps = 1e-3
                bce_real    = -torch.log(real_logits    + eps)
                bce_recon   = -torch.log(1 - recon_logits  + eps)
                bce_sampled = -torch.log(1 - sampled_logits+ eps)

                # Aggregate
                loss_enc = (kl + mse_enc).sum()
                loss_dis = (bce_real + bce_recon + bce_sampled).sum()
                loss_dec = (gamma * mse_dec).sum() - (1 - gamma) * loss_dis

                # Back-prop ----------------------------------------

                # --- Encoder step ---
                model.zero_grad()
                loss_enc.backward(retain_graph=True)
                optimizer_encoder.step()

                # --- Decoder step ---
                model.zero_grad()
                loss_dec.backward(retain_graph=True)
                optimizer_decoder.step()

                # --- Discriminator step ---
                model.zero_grad()
                loss_dis.backward()
                optimizer_discriminator.step()
                    

        '''
        wandb.log({
            "encoder_loss": encoder_loss / batches,
            "decoder_loss": decoder_loss / batches,
            "discriminator_loss": discriminator_loss / batches,
            })
        '''

    return model


if __name__ == '__main__':
    mp.set_start_method('spawn')
    # wandb.login(key=wandb_key)

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

    '''
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
    '''

    train_vaegan(model, optimizer_encoder, optimizer_decoder,
                 optimizer_discriminator, dataloader, dataset, num_epochs)
    # model.load_state_dict(torch.load('model.pth'))

    '''
    # Saving model parameters
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact("model_params", type="model")
    artifact.add_file("model.pth")
    wandb.log_artifact(artifact)

    wandb.finish()
    '''
