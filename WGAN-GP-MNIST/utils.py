import torch

def gradient_penalty(critic, real, fake, labels, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    eps = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated = torch.lerp(real, fake, eps)
    inter_score = critic(interpolated, labels)

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=inter_score,
        grad_outputs=torch.ones_like(inter_score),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
