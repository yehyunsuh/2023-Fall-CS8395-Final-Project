import torch
import torch.nn as nn

from utils import convert_image_to_scalar, convert_scalar_to_image
from model_components import (
    Encoder_CNN, Decoder_CNN,
    Encoder_CVUnet, Decoder_CVUnet
)


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
        image_shape = x.shape
        x = convert_image_to_scalar(x)

        mean, log_sigma_sq = self.encode_x_to_mean_logsigsq(x)
        z = self.encode_mean_logsigsq_to_z(mean, log_sigma_sq)

        output = self.decode_z_to_output(z, lab)
        return convert_scalar_to_image(output, image_shape)

    def forward_train(self, x, lab):
        image_shape = x.shape
        x = convert_image_to_scalar(x)

        mean, log_sigma_sq = self.encode_x_to_mean_logsigsq(x)
        z = self.encode_mean_logsigsq_to_z(mean, log_sigma_sq)

        output = self.decode_z_to_output(z, lab)
        return convert_scalar_to_image(output, image_shape), mean, log_sigma_sq


class CVAE_CNN(nn.Module):
    def __init__(
        self,
        image_shape=(1, 256, 256),
        encoder_kernels=[16, 32, 64, 128, 256],
        decoder_kernels=[64, 32, 16, 8, 4],
        n_z=26,
        n_label=2,
        fc_layers=5
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

        self.n_z = n_z
        self.n_label = n_label
        self.encoder = Encoder_CNN(image_shape, encoder_kernels,
                                   n_z, fc_layers)
        self.decoder = Decoder_CNN(n_z, n_label, image_shape,
                                   decoder_kernels, fc_layers)

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

    # def forward(self, x, lab):
    #     mean, log_sigma_sq = self.encode_x_to_mean_logsigsq(x)
    #     z = self.encode_mean_logsigsq_to_z(mean, log_sigma_sq)
    #     output = self.decode_z_to_output(z, lab)
    #     return output, mean, log_sigma_sq


class CVUnet(nn.Module):
    def __init__(self, image_shape=(1, 256, 256),
                 encoder_kernels=[16, 32, 64, 128, 256],
                 decoder_kernels=[64, 32, 16, 8, 4],
                 n_z=26, n_label=2,
                 fc_layers=5) -> None:
        super(CVUnet, self).__init__()

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

        self.n_z = n_z
        self.n_label = n_label
        self.encoder = Encoder_CVUnet(image_shape, encoder_kernels,
                                      n_z, fc_layers)
        self.decoder = Decoder_CVUnet(n_z, n_label, image_shape,
                                      encoder_kernels, decoder_kernels,
                                      fc_layers)

    def encode_x_to_mean_logsigsq(self, x):
        enc_output, down_outputs = self.encoder(x)
        mean = enc_output[..., :self.n_z]
        log_sigma_sq = enc_output[..., self.n_z:]

        return mean, log_sigma_sq, down_outputs

    def encode_mean_logsigsq_to_z(self, mean, log_sigma_sq):
        sigma = log_sigma_sq.exp().sqrt()
        z = mean + torch.randn_like(mean) * sigma
        return z

    def decode_z_to_output(self, z, lab, down_outputs):
        # concatenate with label
        z_lab = torch.cat((z, lab), axis=-1)

        # decode
        output = self.decoder(z_lab, down_outputs)

        return output

    def forward(self, x, lab):
        mean, log_sigma_sq, down_outputs = self.encode_x_to_mean_logsigsq(x)
        z = self.encode_mean_logsigsq_to_z(mean, log_sigma_sq)
        output = self.decode_z_to_output(z, lab, down_outputs)
        return output

    def forward_train(self, x, lab):
        mean, log_sigma_sq, down_outputs = self.encode_x_to_mean_logsigsq(x)
        z = self.encode_mean_logsigsq_to_z(mean, log_sigma_sq)
        output = self.decode_z_to_output(z, lab, down_outputs)
        return output, mean, log_sigma_sq


def get_model(args):
    if args.model == "CVAE_MLP":
        model = CVAE_MLP(
            args.resize**2,
            args.resize**2,
            5, 
            args.resize,
            n_label=2
        )
    elif args.model == "CVAE_CNN":
        model = CVAE_CNN(
            image_shape=(1, args.resize, args.resize),
            n_z=args.latent_space
        )
    
    # model = nn.DataParallel(model)

    return model


if __name__ == "__main__":
    # encoder_kernels = [2**i for i in range(int(np.log2(512)) + 1)][-5:]
    # print(encoder_kernels)

    x = torch.zeros((10, 1, 256, 256))
    y = torch.zeros((10, 2))

    # cvae = CVAE_CNN(x.shape[1:])
    # print(cvae)
    # print(f"encoded shape: {cvae.encoder(x).shape}")
    # mean, logsigsq = cvae.encode_x_to_mean_logsigsq(x)
    # print(f"mean shape: {mean.shape}; logsigsq shape: {logsigsq.shape}")
    # z = cvae.encode_mean_logsigsq_to_z(mean, logsigsq)
    # print(f"z shape: {z.shape}")
    # output = cvae.decode_z_to_output(z, y)
    # print(f"decoded shape: {output.shape}")
    # print(cvae(x, y).shape)
    # print(cvae)
    NUM_ENC_KERNELS = [19, 41, 97]
    NUM_DEC_KERNELS = [101, 53, 23]
    cvunet = CVUnet(x.shape[1:], NUM_ENC_KERNELS, NUM_DEC_KERNELS)
    print(cvunet)
    print(cvunet(x, y).shape)
