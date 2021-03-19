import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from torch.distributions import kl_divergence

# TODO refactor loss construction to use distributions


class Loss:
    def __init__(self):
        self.log_loss = self.get_loss_dict()

    def get_loss(self, *args, **kwargs):
        return torch.tensor(0)

    def __call__(self, output, target, *args, **kwargs):
        loss = self.get_loss(output, target, *args, **kwargs)
        self.log_loss["loss"] = loss.item()
        return loss

    @classmethod
    def get_loss_dict(cls):
        return Counter({"loss": 0})


def get_mse_loss(output, target, avg_batch=True):
    """
    Reconstruction loss
    To prevent posterior collapse of q(y|x) in GMVAE, there is no normalization performed w.r.t.
    number of frequency bins and time frames; which makes scale of reconstruction loss relatively large
    compared to KL terms.
    TODO:
        [] Allow optional normalization w.r.t. frequency and time axis
        [] Find a good normalization scheme w.r.t frequency and time axis
    """
    output = F.mse_loss(output, target, reduction='none')
    if avg_batch:
        dim_to_sum = list(range(1, len(output.size())))
        output = torch.sum(output, dim=dim_to_sum)
        output = torch.mean(output)
    else:
        output = torch.sum(output)
    return output


class MseLoss(Loss):
    def get_loss(self, output, target, zdist, avg_batch=True):
        return get_mse_loss(output, target)


mse_loss = MseLoss()


class KldGauss(Loss):
    def get_loss(self, output, target, zdist, loc=None, scale=None, avg_batch=True):
        """
        KL divergence between two diagonal Gaussians
        in standard VAEs, the prior p(z) is a standard Gaussian.
        """
        # set prior to a standard Gaussian
        if loc is None:
            loc = torch.zeros_like(zdist.loc)
        if scale is None:
            scale = torch.ones_like(zdist.scale)
        prior = torch.distributions.Normal(loc, scale)
        output = kl_divergence(zdist, prior)
        if avg_batch:
            dim_to_sum = list(range(1, len(output.size())))
            output = torch.sum(output, dim=dim_to_sum)
            output = torch.mean(output)
        else:
            output = torch.sum(output)
        return output


kld_gauss = KldGauss()


class VaeLoss(Loss):
    def get_loss(self, output, target, zdist):
        _mse_loss = mse_loss(output, target, zdist)
        _kld_gauss = kld_gauss(output, target, zdist)
        loss = _mse_loss + _kld_gauss
        self.log_loss["recon_loss"] = _mse_loss.item()
        self.log_loss["kld_loss"] = _kld_gauss.item()
        return loss

    @classmethod
    def get_loss_dict(cls):
        return super().get_loss_dict() + Counter({"recon_loss": 0, "kld_loss": 0})


vae_loss = VaeLoss()
