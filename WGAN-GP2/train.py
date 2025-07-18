import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Critic, Generator, initialize_weights
from utils import gradient_penalty, tests
from tqdm import tqdm
from my_secrets import wandb_key, wandb_proj_name, celeba_img_path, celeba_attr_path
from dataset import CelebAWithText
import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
Z_DIM = 100
ATTR_DIM = 40
NUM_EPOCHS = 500
CRITIC_ITERATIONS = 5
LAMBDA = 10

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(3)],
            [0.5 for _ in range(3)]
        )
    ]
)

dataset = CelebAWithText(celeba_img_path, celeba_attr_path, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

gen = Generator(Z_DIM, ATTR_DIM, IMAGE_SIZE).to(DEVICE)
cri = Critic(ATTR_DIM, IMAGE_SIZE).to(DEVICE)
initialize_weights(gen)
initialize_weights(cri)
opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.9))
opt_cri = optim.Adam(cri.parameters(), lr=LR, betas=(0.0, 0.9))
gen.train()
cri.train()

wandb.login(key=wandb_key)
wandb.init(project=wandb_proj_name, config={"model": "WGAN-GP", "epochs": NUM_EPOCHS})

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, attr) in enumerate(tqdm(loader, desc=f"Epoch: {epoch}")):
        real = real.to(DEVICE)
        attr0 = attr[0].to(DEVICE)
        attr1 = attr[1].to(DEVICE)
        noise = torch.randn((real.size(0), Z_DIM, 1, 1)).to(DEVICE)
        fake = gen(noise, attr0)
        cri_fake = cri(fake, attr1).reshape(-1)

        if batch_idx % (CRITIC_ITERATIONS + 1) == 0: 
            loss_gen = -torch.mean(cri_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
        
        else:
            cri_real = cri(real, attr1).reshape(-1)
            gp = gradient_penalty(cri, real, fake, attr1, DEVICE)
            loss_cri = torch.mean(cri_fake) - torch.mean(cri_real) + LAMBDA * gp
            cri.zero_grad()
            loss_cri.backward()
            opt_cri.step()

        if batch_idx > 0 and batch_idx % 300 == 0:
            with torch.no_grad():
                images = []
                for test in tests:
                    z = torch.randn((1, Z_DIM, 1, 1)).to(DEVICE)
                    attr = test[0].unsqueeze(0).to(DEVICE)
                    fake_image = gen(z, attr).detach().cpu()
                    images.append(wandb.Image(fake_image[0], caption=test[1]))
                wandb.log({"Generated Images": images})
                wandb.log({"Generator Loss": loss_gen.item(), "Critic Loss": loss_cri.item()})
    
    torch.save({
        'generator_state_dict': gen.state_dict(),
        'critic_state_dict': cri.state_dict(),
        'g_opt_state_dict': opt_gen.state_dict(),
        'c_opt_state_dict': opt_cri.state_dict(),
    }, 'checkpoint2.pth')
    wandb.save('checkpoint2.pth')

wandb.finish()
