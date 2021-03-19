import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super().__init__()
        self.model_device = "cpu"

    @abstractmethod
    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + '\nTrainable parameters: {}'.format(params)

    def to(self, device, *args, **kwargs):
        self.model_device = device
        return super().to(device=device, *args, **kwargs)


class BaseVAE(BaseModel):
    def __init__(self, input_size, latent_dim):
        super(BaseVAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

    def infer_flat_size(self):
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def _infer_latent(self, logits):
        zdist = generate_gaussian(logits)
        return zdist

    def forward(self, x):
        raise NotImplementedError


def generate_gaussian(logits):
    mu, sigma = torch.chunk(logits, chunks=2, dim=1)
    sigma = F.softplus(sigma)
    zdist = distributions.Normal(loc=mu, scale=sigma)
    return zdist
