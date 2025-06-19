from torch.utils.data import Dataset
import torch
import wandb

# prompts to test model generation
prompts = [
    "A young attractive woman with bangs, wearing eyeglasses",
    "Old black bald man with beard",
    "Young guy with big smile, big lips and nose and black hair",
    "Female with heavy makeup and high cheekbones",
    "Man wearing necktie with wavy blond hair and sideburns",
    "Old man with receding hairline, pale skin and pointy nose",
    "Male with big nose, bangs and a mustache",
]

# indexes to test model reconstruction (only VAE)
recon_indexes = [0, 120, 69, 2137, 420, 6969, 10000]


def generate_from_prompts(model):
    images = [wandb.Image(model.generate_image(prompt), caption=prompt)
              for prompt in prompts]
    wandb.log({"Generated image": images})


def reconstruct_images(model, dataset, device):
    imgs, labels, clip_embs = [], [], []
    for idx in recon_indexes:
        img, lbl, clip_e = dataset[idx]
        imgs.append(img)
        labels.append(lbl)
        clip_embs.append(clip_e)

    x_orig = torch.stack(imgs, dim=0).to(device)
    labels = torch.stack(labels, dim=0).to(device)
    clip_emb = torch.stack(clip_embs, dim=0).to(device)

    with torch.no_grad():
        model.eval()
        x_tilde, *_ = model(x_orig, labels, clip_emb)

    return x_orig, x_tilde, labels, clip_emb


def log_reconstructed_images(model, dataset, device):
    x_real, x_recon, _, _ = reconstruct_images(model, dataset, device)

    recon_imgs = []
    for real, recon in zip(x_real, x_recon):
        # real / recon are tensors in [-1,1] or [0,1], adapt as needed:
        recon = (recon.clamp(0, 1) + 1) * 127.5
        real = (real.clamp(0, 1) + 1) * 127.5
        recon = recon.cpu().permute(1, 2, 0).byte().numpy()
        real = real.cpu().permute(1, 2, 0).byte().numpy()
        recon_imgs.append({
          "original": wandb.Image(real),
          "reconstruction": wandb.Image(recon)
        })

    wandb.log({"reconstructions": recon_imgs})
