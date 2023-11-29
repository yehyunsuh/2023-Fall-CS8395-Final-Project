import torch
import torch.nn as nn

from functools import reduce


class Encoder_CNN(nn.Module):
    def __init__(self, input_size: (1, 256, 256),
                 num_kernels=[16, 32, 64, 128, 256], n_z=26):
        super(Encoder_CNN, self).__init__()

        self.enc = nn.ModuleList()
        for i in range(len(num_kernels)):
            if i == 0:
                self.enc.append(nn.Conv2d(input_size[0],
                                          num_kernels[i], 3, 1, 1))
            else:
                self.enc.append(nn.Conv2d(num_kernels[i-1],
                                          num_kernels[i], 3, 1, 1))
            self.enc.append(nn.BatchNorm2d(num_kernels[i]))
            self.enc.append(nn.ReLU())

            self.enc.append(nn.Conv2d(num_kernels[i], num_kernels[i], 3, 1, 1))
            self.enc.append(nn.BatchNorm2d(num_kernels[i]))
            self.enc.append(nn.ReLU())

            self.enc.append(nn.Conv2d(num_kernels[i], num_kernels[i], 3, 1, 1))
            self.enc.append(nn.BatchNorm2d(num_kernels[i]))
            self.enc.append(nn.ReLU())

            self.enc.append(nn.MaxPool2d(2, 2))

        # self.enc.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.enc.append(nn.Flatten())
        linear_input_size = (
            num_kernels[-1] *
            (input_size[1]//(2**len(num_kernels))) *
            (input_size[2]//(2**len(num_kernels)))
        )
        self.enc.append(nn.Linear(linear_input_size, 2 * n_z))

        self.enc = nn.Sequential(*self.enc)

    def forward(self, x):
        return self.enc(x.float())


class Decoder_CNN(nn.Module):
    def __init__(self, n_z=26, n_label=2,
                 output_size=(1, 256, 256),
                 num_kernels=[64, 32, 16, 8, 4]):
        super(Decoder_CNN, self).__init__()

        self.dec = nn.ModuleList()
        unflatten_shape = (
            num_kernels[0],
            output_size[1]//(2**len(num_kernels)),
            output_size[2]//(2**len(num_kernels))
        )
        unflatten_shape_prod = reduce(lambda x, y: x * y, unflatten_shape)
        self.dec.append(nn.Linear(n_z + n_label, unflatten_shape_prod))
        self.dec.append(nn.ReLU())
        self.dec.append(nn.Unflatten(1, unflatten_shape))
        for i in range(len(num_kernels)-1):
            self.dec.append(nn.Conv2d(num_kernels[i], num_kernels[i], 3, 1, 1))
            self.dec.append(nn.BatchNorm2d(num_kernels[i]))
            self.dec.append(nn.ReLU())

            self.dec.append(nn.Conv2d(num_kernels[i], num_kernels[i], 3, 1, 1))
            self.dec.append(nn.BatchNorm2d(num_kernels[i]))
            self.dec.append(nn.ReLU())

            self.dec.append(nn.ConvTranspose2d(num_kernels[i],
                                               num_kernels[i+1], 3, 2, 1, 1))
            self.dec.append(nn.BatchNorm2d(num_kernels[i+1]))
            self.dec.append(nn.ReLU())

        self.dec.append(nn.ConvTranspose2d(num_kernels[-1],
                                           output_size[0], 3, 2, 1, 1))
        self.dec.append(nn.BatchNorm2d(output_size[0]))
        self.dec.append(nn.Sigmoid())
        self.dec = nn.Sequential(*self.dec)

    def forward(self, x):
        return self.dec(x.float())


class CVAE_CNN(nn.Module):
    def __init__(
        self,
        image_shape=(1, 256, 256),
        encoder_kernels=[16, 32, 64, 128, 256],
        decoder_kernels=[64, 32, 16, 8, 4],
        n_z=26,
        n_label=2
    ):
        super(CVAE_CNN, self).__init__()

        assert len(encoder_kernels) == len(decoder_kernels), \
            "encoder and decoder must have the same number of layers"\
            f"got {len(encoder_kernels)} and {len(decoder_kernels)}"

        assert image_shape[1] % (2**len(encoder_kernels)) == 0, \
            "image size must be divisible by 2**len(encoder_kernels)"\
            f"got {image_shape[1]} and {2**len(encoder_kernels)}"

        assert image_shape[2] % (2**len(encoder_kernels)) == 0, \
            "image size must be divisible by 2**len(encoder_kernels)"\
            f"got {image_shape[2]} and {2**len(encoder_kernels)}"

        assert image_shape[1] % (2**len(decoder_kernels)) == 0, \
            "image size must be divisible by 2**len(decoder_kernels)"\
            f"got {image_shape[1]} and {2**len(decoder_kernels)}"

        assert image_shape[2] % (2**len(decoder_kernels)) == 0, \
            "image size must be divisible by 2**len(decoder_kernels)"\
            f"got {image_shape[2]} and {2**len(decoder_kernels)}"

        assert n_z % 2 == 0, \
            "n_z must be divisible by 2"\
            f"got {n_z}"

        self.n_z = n_z
        self.n_label = n_label
        self.encoder = Encoder_CNN(image_shape, encoder_kernels, n_z)
        self.decoder = Decoder_CNN(n_z, n_label, image_shape, decoder_kernels)

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
    model = CVAE_CNN(
        image_shape=[1, 32, 32], 
        n_label=10
    )

    return model