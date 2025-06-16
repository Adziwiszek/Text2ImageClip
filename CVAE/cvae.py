import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, CelebA
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
import os
import zipfile
import requests
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
import typer

from my_secrets import wandb_proj_name, wandb_key, img_path, attr_path

device = 'cuda'

# Map to turn binary attributes to something that resembles a sentence
template_map = {
    "5_o_Clock_Shadow": ("with a 5 o'clock shadow", ""),
    "Arched_Eyebrows": ("with arched eyebrows", ""),
    "Attractive": ("looking attractive", ""),
    "Bags_Under_Eyes": ("with bags under the eyes", ""),
    "Bald": ("who is bald", ""),
    "Bangs": ("with bangs", ""),
    "Big_Lips": ("with big lips", ""),
    "Big_Nose": ("with a big nose", ""),
    "Black_Hair": ("with black hair", ""),
    "Blond_Hair": ("with blond hair", ""),
    "Blurry": ("in a blurry photo", ""),
    "Brown_Hair": ("with brown hair", ""),
    "Bushy_Eyebrows": ("with bushy eyebrows", ""),
    "Chubby": ("who is chubby", ""),
    "Double_Chin": ("with a double chin", ""),
    "Eyeglasses": ("wearing eyeglasses", ""),
    "Goatee": ("with a goatee", ""),
    "Gray_Hair": ("with gray hair", ""),
    "Heavy_Makeup": ("wearing heavy makeup", ""),
    "High_Cheekbones": ("with high cheekbones", ""),
    "Male": ("man", "woman"),
    "Mouth_Slightly_Open": ("with mouth slightly open", ""),
    "Mustache": ("with a mustache", ""),
    "Narrow_Eyes": ("with narrow eyes", ""),
    "No_Beard": ("without a beard", ""),
    "Oval_Face": ("with an oval face", ""),
    "Pale_Skin": ("with pale skin", ""),
    "Pointy_Nose": ("with a pointy nose", ""),
    "Receding_Hairline": ("with a receding hairline", ""),
    "Rosy_Cheeks": ("with rosy cheeks", ""),
    "Sideburns": ("with sideburns", ""),
    "Smiling": ("smiling", ""),
    "Straight_Hair": ("with straight hair", ""),
    "Wavy_Hair": ("with wavy hair", ""),
    "Wearing_Earrings": ("wearing earrings", ""),
    "Wearing_Hat": ("wearing a hat", ""),
    "Wearing_Lipstick": ("wearing lipstick", ""),
    "Wearing_Necklace": ("wearing a necklace", ""),
    "Wearing_Necktie": ("wearing a necktie", ""),
    "Young": ("a young", "an old"),
}

# Prompts for generating images each epoch
prompts = [
    "A bald woman with glasses",
    "A young attractive woman with bangs, wearing glasses",
    "Old black bald man with beard",
]

def generate_and_log_images(model):
    images = [wandb.Image(model.generate_image(prompt), caption=prompt) for prompt in prompts]
    wandb.log({"Generated image": images})

def generate_text_embeddings(text_prompts, clip_model):
    text_tokens = clip.tokenize(text_prompts).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.encode_text(text_tokens)
    return text_embeddings.cpu()

def attributes_to_sentence(attr_vector, attributes):
    # Build sentence components
    active = [name for bit, name in zip(attr_vector, attributes) if bit == 1]

    gender = template_map['Male'][0] if 'Male' in active else template_map['Male'][1]
    age = template_map['Young'][0] if 'Young' in active else template_map['Young'][1]

    description_parts = [age, gender]

    for attr in active:
        if attr in template_map and attr not in ('Male', 'Young'):
            phrase = template_map[attr][0]
            if phrase:
                description_parts.append(phrase)

    return " ".join(description_parts)

class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_df, attributes, clip_model, transform=None):
        self.img_dir = img_dir
        self.attr_df = attr_df
        self.transform = transform
        self.attributes = attributes
        self.clip_model = clip_model

    def __len__(self):
        return len(self.attr_df)

    def __getitem__(self, idx):
        filename = self.attr_df.iloc[idx].image_id
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        row = self.attr_df.iloc[idx]
        label = torch.tensor(row[1:].values.astype(float),
                dtype=torch.float32)
        label = label + 1
        label = label // 2

        clip_embed = generate_text_embeddings(attributes_to_sentence(label, self.attributes), self.clip_model)

        return image, label, clip_embed

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
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(512*4*4, latent_dim)
        self.fc_logvar = nn.Linear(512*4*4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + cond_dim + clip_dim, 512*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
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

            # Step 2: Sample latent vector z ~ N(0, I)
            z = torch.randn(1, self.latent_dim).to(device)

            # Step 3: Dummy condition vector (e.g., zeros if not specified)
            if c is None:
                c = torch.zeros(1, self.cond_dim).to(device)

            # Step 4: Decode to image
            recon = self.decode(z, c, text_embedding)

            # Output shape: [1, 3, H, W], scale from [-1, 1] to [0, 255]
            recon = recon.squeeze(0).detach().cpu()
            recon = (recon + 1) / 2  # scale to [0, 1]
            recon = recon.clamp(0, 1)
            recon_img = (recon * 255).byte().permute(1, 2, 0).numpy()

            return Image.fromarray(recon_img)

def loss_function(recon_x, x, mu, logvar, criterion):
    recon_loss = criterion(recon_x, x)
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KL

def train(model, dataloader, optimizer, criterion, num_epochs):
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, (data, attrs, clip_emb) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
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
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    for param in clip_model.parameters():
        param.requires_grad = False

    # Loading image attributes
    df_attrs = pd.read_csv(attr_path)

    attributes = [col for col in df_attrs.columns]
    attribute2id = {att: id for id, att in enumerate(attributes)}
    id2attribute = {id: att for id, att in enumerate(attributes)}

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = CelebADataset(
        img_dir=img_path,
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
            "epochs":num_epochs,
            "batch_size":batch_size,
            "learning_rate":lr,
            "fresh_start": False,
            "batch_norm": False,
            }
    )

    train(model, dataloader, optimizer, criterion, num_epochs)
    #model.load_state_dict(torch.load('model.pth'))

    # Saving model parameters
    #torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact("model_params", type="model")
    artifact.add_file("model.pth")
    wandb.log_artifact(artifact)

    # Generating images on trained model and logging them on wandb

    wandb.finish()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    #app()
    run_training()
