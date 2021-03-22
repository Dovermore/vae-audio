import numpy as np
import torch
import torch.nn as nn
from base import BaseVAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_activation_module(activation):
    if activation == "leaky_relu":
        activation = nn.LeakyReLU
    elif activation == "tanh":
        activation = nn.Tanh
    elif activation is None:
        activation = nn.Identity
    else:
        raise ValueError("Activation must be one of 'leaky_relu', 'tanh'")
    return activation


def spec_conv1d(n_layer=3, n_channel=[64, 32, 16, 8], filter_size=[1, 3, 3], stride=[1, 2, 2], activation="leaky_relu"):
    """
    Construction of conv. layers. Note the current implementation always effectively turn to 1-D conv,
    inspired by https://arxiv.org/pdf/1704.04222.pdf.
    :param n_layer: number of conv. layers
    :param n_channel: in/output number of channels for each layer ( len(n_channel) = n_layer + 1 ).
            The first channel is the number of freqeuncy bands of input spectrograms
    :param filter_size: the filter size (x-axis) for each layer ( len(filter_size) = n_layer )
    :param stride: filter stride size (x-axis) for each layer ( len(stride) = n_layer )
    :param activation: activation function to use
    :return: an object (nn.Sequential) constructed of specified conv. layers
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    ast_msg = "The following must fulfill: len(filter_size) == len(stride) == n_layer"
    assert len(filter_size) == len(stride) == n_layer, ast_msg
    activation_module = get_activation_module(activation)

    # construct layers
    conv_layers = []
    for i in range(n_layer):
        in_channel, out_channel = n_channel[i:i + 2]
        conv_layers += [
            nn.Conv1d(in_channel, out_channel, filter_size[i], stride[i]),
            nn.BatchNorm1d(out_channel),
            activation_module()
        ]
    return nn.Sequential(*conv_layers)


def spec_deconv1d(n_layer=3, n_channel=[64, 32, 16, 8], filter_size=[1, 3, 3], stride=[1, 2, 2], activation="leaky_relu"):
    """
    Construction of deconv. layers. Input the arguments in normal conv. order.
    E.g., n_channel = [1, 32, 16, 8] gives deconv. layers of [8, 16, 32, 1].
    :param n_layer: number of deconv. layers
    :param n_channel: in/output number of channels for each layer ( len(n_channel) = n_layer + 1 )
            The first channel is the number of freqeuncy bands of input spectrograms
    :param filter_size: the filter size (x-axis) for each layer ( len(filter_size) = n_layer )
    :param stride: filter stride size (x-axis) for each layer ( len(stride) = n_layer )
    :return: an object (nn.Sequential) constructed of specified deconv. layers.
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    ast_msg = "The following must fulfill: len(filter_size) == len(stride) == n_layer"
    assert len(filter_size) == len(stride) == n_layer, ast_msg
    activation_module = get_activation_module(activation)

    n_channel, filter_size, stride = n_channel[::-1], filter_size[::-1], stride[::-1]

    deconv_layers = []
    for i in range(n_layer - 1):
        in_channel, out_channel = n_channel[i:i + 2]
        deconv_layers += [
            nn.ConvTranspose1d(in_channel, out_channel, filter_size[i], stride[i]),
            nn.BatchNorm1d(out_channel),
            activation_module()
        ]

    # Construct the output layer
    deconv_layers += [
        nn.ConvTranspose1d(n_channel[-2], n_channel[-1], filter_size[-1], stride[-1]),
        activation_module()
    ]

    return nn.Sequential(*deconv_layers)


def fc(n_layer, n_channel, activation='leaky_relu', batchNorm=True):
    """
    Construction of fc. layers.
    :param n_layer: number of fc. layers
    :param n_channel: in/output number of neurons for each layer ( len(n_channel) = n_layer + 1 )
    :param activation: allow either 'tanh' or None for now
    :param batchNorm: True|False, indicate apply batch normalization or not
    """

    assert len(n_channel) == n_layer + 1, "This must fulfill: len(n_channel) = n_layer + 1"
    activation_module = get_activation_module(activation)

    fc_layers = []
    for i in range(n_layer):
        layer = [nn.Linear(n_channel[i], n_channel[i + 1])]
        if batchNorm:
            layer.append(nn.BatchNorm1d(n_channel[i + 1]))
        if activation:
            layer.append(activation_module())
        fc_layers += layer

    return nn.Sequential(*fc_layers)


class SpecVAE(BaseVAE):
    def __init__(self, input_size=(64, 15), latent_dim=32,
                 n_convLayer=3, n_convChannel=[32, 16, 8], filter_size=[1, 3, 3], stride=[1, 2, 2],
                 n_fcLayer=1, n_fcChannel=[256], activation="leaky_relu"):
        """
        Construction of VAE
        :param input_size: (n_freqBand, n_contextWin);
                           assume a spectrogram input of size (n_freqBand, n_contextWin)
        :param latent_dim: the dimension of the latent vector
        :param is_featExtract: if True, output z as mu; otherwise, output z derived from reparameterization trick
        """
        super(SpecVAE, self).__init__(input_size, latent_dim)
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.activation = activation

        self.n_freqBand, self.n_contextWin = input_size

        # Construct encoder and Gaussian layers
        self.encoder_conv = spec_conv1d(n_convLayer, [self.n_freqBand] + n_convChannel, filter_size, stride)
        self.flat_size, self.encoder_outputSize = self._infer_flat_size()
        self.encoder_fc = fc(n_fcLayer, [self.flat_size, *n_fcChannel], activation=activation, batchNorm=True)
        self.encoder_fc = nn.Sequential(*self.encoder_fc, fc(1, [n_fcChannel[-1], latent_dim * 2],
                                                             activation=None, batchNorm=False))
        # Construct decoder
        self.decoder_fc = fc(n_fcLayer + 1, [self.latent_dim, *n_fcChannel[::-1], self.flat_size],
                             activation=activation, batchNorm=True)
        self.decoder = spec_deconv1d(n_convLayer, [self.n_freqBand] + n_convChannel, filter_size, stride)

    def _infer_flat_size(self):
        encoder_output = self.encoder_conv(torch.ones(1, *self.input_size))
        return int(np.prod(encoder_output.size()[1:])), encoder_output.size()[1:]

    def encode(self, x):
        if len(x.shape) == 4:
            assert x.shape[1] == 1
            x = x.squeeze(1)

        logits = self.encoder_conv(x)
        logits = self.encoder_fc(logits.view(-1, self.flat_size))
        zdist = self._infer_latent(logits)
        return zdist

    def decode(self, z):
        h = self.decoder_fc(z)
        x_recon = self.decoder(h.view(-1, *self.encoder_outputSize))
        return x_recon

    def forward(self, x):
        zdist = self.encode(x)
        z = zdist.rsample()
        x_recon = self.decode(z)
        return x_recon, zdist

    def sample(self, z=None):
        if z is None:
            logits = torch.zeros([[self.latent_dim * 2]]).to(self.model_device)
            zdist = self._infer_latent(logits)
            z = zdist.rsample()
        x_gen = self.decode(z)
        return x_gen