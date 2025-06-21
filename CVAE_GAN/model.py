import torch

from CVAE.model import ClipCVAE
from .discriminator import Discriminator


class CVAE_GAN(ClipCVAE):
    def __init__(self, clip_model):
        super(CVAE_GAN, self).__init__(clip_model)
        self.clip_model = clip_model
        self.discriminator = Discriminator()

    def forward(self, x, c, clip_embedding):
        return super().forward(x, c, clip_embedding)

    @staticmethod
    def loss(disc_feats_original, disc_feats_predicted, disc_feats_sampled,
             disc_class_original, disc_class_predicted, disc_class_sampled,
             mus, variances):
        # kl-divergence
        kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1, 1)

        # mse between intermediate featss
        mse = torch.sum(0.5*(disc_feats_original - disc_feats_predicted) ** 2, 1)

        # bce for decoder and discriminator for original and reconstructed
        bce_dis_original = -torch.log(disc_class_original + 1e-3)
        bce_dis_predicted = -torch.log(1 - disc_class_predicted + 1e-3)
        bce_dis_sampled = -torch.log(1 - disc_class_sampled + 1e-3)

        return kl, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled
