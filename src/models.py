"""Currently, CVAE_CNN is still hard-coded for 256x256 images.
The encoder and decoder need to have the same number of layers.
TODO: make it more flexible?
"""

import torch
import torch.nn as nn


class CVAE_MLP(nn.Module):
    def __init__(self, n_input: int, n_output: int,
                 n_layers: int, layer_size: int,
                 n_z: int = 2, n_label: int = 10) -> None:
        super(CVAE_MLP, self).__init__()

        self.n_z = n_z
        self.n_label = n_label

        self.enc = nn.ModuleList()
        self.enc.append(nn.Linear(n_input, layer_size))
        self.enc.append(nn.ReLU())

        self.dec = nn.ModuleList()
        self.dec.append(nn.Linear(self.n_z + self.n_label, layer_size))
        self.dec.append(nn.ReLU())

        for i in range(n_layers):
            self.enc.append(nn.Linear(layer_size, layer_size))
            self.enc.append(nn.ReLU())

            self.dec.append(nn.Linear(layer_size, layer_size))
            self.dec.append(nn.ReLU())

        # we need this for enc's mu and sigma
        self.enc.append(nn.Linear(layer_size, 2 * self.n_z))

        self.dec.append(nn.Linear(layer_size, n_output))

        self.enc = nn.Sequential(*self.enc)
        self.dec = nn.Sequential(*self.dec)

    def encode_x_to_mean_logsigsq(self, x):
        enc_output = self.enc(x)

        # the first n_z are assigned to mean
        mean = enc_output[..., :self.n_z]

        # the other n_z are assigned to log-sigma-squared
        log_sigma_sq = enc_output[..., self.n_z:]

        return mean, log_sigma_sq

    def encode_mean_logsigsq_to_z(self, mean, log_sigma_sq):
        sigma = log_sigma_sq.exp().sqrt()
        z = mean + torch.randn_like(mean) * sigma
        return z

    def decode_z_to_output(self, z, lab):
        z_lab = torch.cat((z, lab), axis=-1)

        return self.dec(z_lab)

    def forward(self, x, lab):
        mean, log_sigma_sq = self.encode_x_to_mean_logsigsq(x)
        z = self.encode_mean_logsigsq_to_z(mean, log_sigma_sq)
        output = self.decode_z_to_output(z, lab)
        return output

    def forward_train(self, x, lab):
        mean, log_sigma_sq = self.encode_x_to_mean_logsigsq(x)
        z = self.encode_mean_logsigsq_to_z(mean, log_sigma_sq)
        output = self.decode_z_to_output(z, lab)
        return output, mean, log_sigma_sq


class Encoder_CNN(nn.Module):
    def __init__(self, input_channels: int = 1,
                 num_kernels=[16, 32, 64, 128, 256]):
        super(Encoder_CNN, self).__init__()

        self.enc = nn.ModuleList()
        for i in range(len(num_kernels)):
            if i == 0:
                self.enc.append(nn.Conv2d(input_channels,
                                          num_kernels[i], 3, 1, 1))
            else:
                self.enc.append(nn.Conv2d(num_kernels[i-1],
                                          num_kernels[i], 3, 1, 1))
            self.enc.append(nn.ReLU())
            self.enc.append(nn.MaxPool2d(2))
        self.enc.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.enc.append(nn.Flatten())

        self.enc = nn.Sequential(*self.enc)

    def forward(self, x):
        return self.enc(x.float())


class Decoder_CNN(nn.Module):
    def __init__(self, input_channels: int,
                 num_kernels=[64, 32, 16, 8, 4]):
        super(Decoder_CNN, self).__init__()

        self.dec = nn.ModuleList()
        self.dec.append(nn.Unflatten(1, (input_channels, 1, 1)))
        self.dec.append(nn.AdaptiveAvgPool2d((8, 8)))  # TODO: change this
        for i in range(len(num_kernels)):
            if i == 0:
                self.dec.append(nn.ConvTranspose2d(input_channels,
                                                   num_kernels[i], 3, 2, 1, 1))
            else:
                self.dec.append(nn.ConvTranspose2d(num_kernels[i-1],
                                                   num_kernels[i], 3, 2, 1, 1))
            self.dec.append(nn.ReLU())

        self.dec.append(nn.Conv2d(num_kernels[-1], 1, 3, 1, 1))
        self.dec.append(nn.Sigmoid())
        self.dec = nn.Sequential(*self.dec)

    def forward(self, x):
        return self.dec(x.float())


class CVAE_CNN(nn.Module):
    def __init__(self, input_channels=1,
                 encoder_kernels=[16, 32, 64, 128, 256],
                 decoder_kernels=[64, 32, 16, 8, 4],
                 n_label=2):
        super(CVAE_CNN, self).__init__()

        self.n_z = encoder_kernels[-1]//2
        self.n_label = n_label
        self.encoder = Encoder_CNN(input_channels, encoder_kernels)
        self.decoder = Decoder_CNN(encoder_kernels[-1]//2 + n_label,
                                   decoder_kernels)

    def encode_x_to_mean_logsigsq(self, x):
        enc_output = self.encoder(x)

        # the first n_z are assigned to mean
        mean = enc_output[..., :self.n_z]

        # the other n_z are assigned to log-sigma-squared
        log_sigma_sq = enc_output[..., self.n_z:]

        return mean, log_sigma_sq

    def encode_mean_logsigsq_to_z(self, mean, log_sigma_sq):
        sigma = log_sigma_sq.exp().sqrt()
        z = mean + torch.randn_like(mean) * sigma
        return z

    def decode_z_to_output(self, z, lab):
        z_lab = torch.cat((z, lab), axis=-1)
        return self.decoder(z_lab)

    def forward(self, x, lab):
        mean, log_sigma_sq = self.encode_x_to_mean_logsigsq(x)
        z = self.encode_mean_logsigsq_to_z(mean, log_sigma_sq)
        output = self.decode_z_to_output(z, lab)
        return output

    def forward_train(self, x, lab):
        mean, log_sigma_sq = self.encode_x_to_mean_logsigsq(x)
        z = self.encode_mean_logsigsq_to_z(mean, log_sigma_sq)
        output = self.decode_z_to_output(z, lab)
        return output, mean, log_sigma_sq


def get_model(args):
    if args.model == "CVAE_MLP":
        model = CVAE_MLP()
    elif args.model == "CVAE_CNN":
        model = CVAE_CNN()

    return model


if __name__ == "__main__":
    x = torch.zeros((10, 3, 256, 256))
    y = torch.zeros((10, 2))
    cvae = CVAE_CNN()
    print(cvae(x, y).shape)
    # enc = Encoder_CNN(1)
    # z = enc(x)
    # print(z.shape)
    # dec = Decoder_CNN(256)
    # print(dec(z).shape)
