import torch

tests = [([-1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0], 'a young woman with arched eyebrows looking attractive with brown hair wearing heavy makeup with high cheekbones with mouth slightly open without a beard with a pointy nose smiling with straight hair wearing earrings wearing lipstick'),
([-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0], 'a young woman with bags under the eyes with bangs with blond hair with high cheekbones with mouth slightly open without a beard with a pointy nose smiling'),
([-1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0], 'a young woman with arched eyebrows looking attractive with blond hair wearing heavy makeup without a beard with an oval face with pale skin wearing lipstick wearing a necklace')]

tests = [(torch.tensor(test[0], dtype=torch.float32),test[1]) for test in tests]

def gradient_penalty(critic, real, fake, attr, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    eps = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated = torch.lerp(real, fake, eps)
    inter_score = critic(interpolated, attr)

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
