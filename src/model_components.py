from functools import reduce

import torch
import torch.nn as nn


class Encoder_CNN(nn.Module):
    def __init__(self, input_size: (1, 256, 256),
                 num_kernels=[16, 32, 64, 128, 256],
                 n_z=26, n_fc_layers=5):
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
        down_size_ratio = int((linear_input_size / n_z)**(1/(1+n_fc_layers)))

        for i in range(n_fc_layers):
            linear_output_size = linear_input_size//down_size_ratio
            self.enc.append(nn.Linear(linear_input_size, linear_output_size))
            self.enc.append(nn.BatchNorm1d(linear_output_size))
            self.enc.append(nn.ReLU())
            linear_input_size = linear_output_size

        self.enc.append(nn.Linear(linear_input_size, 2 * n_z))
        self.enc.append(nn.BatchNorm1d(2 * n_z))
        self.enc.append(nn.ReLU())

        self.enc = nn.Sequential(*self.enc)

    def forward(self, x):
        return self.enc(x.float())


class Decoder_CNN(nn.Module):
    def __init__(self, n_z=26, n_label=2,
                 output_size=(1, 256, 256),
                 num_kernels=[64, 32, 16, 8, 4],
                 n_fc_layers=5):
        super(Decoder_CNN, self).__init__()

        self.dec = nn.ModuleList()
        unflatten_shape = (
            num_kernels[0],
            output_size[1]//(2**len(num_kernels)),
            output_size[2]//(2**len(num_kernels))
        )
        unflatten_shape_prod = reduce(lambda x, y: x * y, unflatten_shape)
        linear_input_size = n_z + n_label
        up_size_ratio = int((unflatten_shape_prod / linear_input_size) **
                            (1 / (1 + n_fc_layers)))
        print(f"Up size ratio: {up_size_ratio}")
        for i in range(n_fc_layers):
            linear_output_size = linear_input_size * up_size_ratio
            self.dec.append(nn.Linear(linear_input_size, linear_output_size))
            self.dec.append(nn.BatchNorm1d(linear_output_size))
            self.dec.append(nn.ReLU())
            linear_input_size = linear_output_size

        self.dec.append(nn.Linear(linear_input_size, unflatten_shape_prod))
        self.dec.append(nn.BatchNorm1d(unflatten_shape_prod))
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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1) -> None:
        super(DoubleConv, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(in_channels, out_channels,
                                   kernel_size, padding=padding))
        self.conv.append(nn.BatchNorm2d(out_channels))
        self.conv.append(nn.ReLU(inplace=True))
        self.conv.append(nn.Conv2d(out_channels, out_channels,
                                   kernel_size, padding=padding))
        self.conv.append(nn.BatchNorm2d(out_channels))
        self.conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1) -> None:
        super(DownBlock, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(nn.MaxPool2d(2, 2))
        self.conv.append(DoubleConv(in_channels, out_channels,
                                    kernel_size, padding))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1) -> None:
        super(UpBlock, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=True))
        self.conv.append(DoubleConv(in_channels, out_channels,
                                    kernel_size, padding))
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)


class Encoder_CVUnet(nn.Module):
    def __init__(self, image_size: (1, 256, 256),
                 num_enc_kernels=[16, 32, 64, 128, 256],
                 n_z=26, n_fc_layers=5):
        super(Encoder_CVUnet, self).__init__()

        self.enc = nn.ModuleList()
        self.enc.append(DoubleConv(image_size[0], num_enc_kernels[0]))
        for i in range(len(num_enc_kernels)):
            if i == 0:
                self.enc.append(DownBlock(num_enc_kernels[i],
                                          num_enc_kernels[i]))
            else:
                self.enc.append(DownBlock(num_enc_kernels[i-1],
                                          num_enc_kernels[i]))

        self.fc = nn.ModuleList()
        self.fc.append(nn.Flatten())

        linear_input_size = (
            num_enc_kernels[-1] *
            (image_size[1]//(2**len(num_enc_kernels))) *
            (image_size[2]//(2**len(num_enc_kernels)))
        )
        down_size_ratio = int((linear_input_size/(2*n_z))**(1/(1+n_fc_layers)))

        for i in range(n_fc_layers):
            linear_output_size = linear_input_size // down_size_ratio
            self.fc.append(nn.Linear(linear_input_size, linear_output_size))
            self.fc.append(nn.BatchNorm1d(linear_output_size))
            self.fc.append(nn.ReLU())
            linear_input_size = linear_output_size

        self.fc.append(nn.Linear(linear_input_size, 2 * n_z))
        self.fc.append(nn.BatchNorm1d(2 * n_z))
        self.fc.append(nn.ReLU())

    def forward(self, x):
        output = x
        outputs = []
        for layer in self.enc:
            output = layer(output)
            outputs.append(output)
        for layer in self.fc:
            output = layer(output)
        return output, outputs


class Decoder_CVUnet(nn.Module):
    def __init__(self, n_z=26, n_label=2,
                 image_size=(1, 256, 256),
                 num_enc_kernels=[16, 32, 64, 128, 256],
                 num_dec_kernels=[64, 32, 16, 8, 4],
                 n_fc_layers=5):
        super(Decoder_CVUnet, self).__init__()

        assert len(num_enc_kernels) == len(num_dec_kernels), \
            "Number of encoder and decoder kernels must be the same"\
            f"got {len(num_enc_kernels)} and {len(num_dec_kernels)}"

        self.fc = nn.ModuleList()
        linear_input_size = n_z + n_label
        unflatten_shape = (
            num_dec_kernels[0],
            image_size[1]//(2**len(num_dec_kernels)),
            image_size[2]//(2**len(num_dec_kernels))
        )
        unflatten_shape_prod = reduce(lambda x, y: x * y, unflatten_shape)
        up_size_ratio = int((unflatten_shape_prod / linear_input_size) **
                            (1 / (1 + n_fc_layers)))
        for i in range(n_fc_layers):
            linear_output_size = linear_input_size * up_size_ratio
            self.fc.append(nn.Linear(linear_input_size, linear_output_size))
            self.fc.append(nn.BatchNorm1d(linear_output_size))
            self.fc.append(nn.ReLU())
            linear_input_size = linear_output_size

        self.fc.append(nn.Linear(linear_input_size, unflatten_shape_prod))
        self.fc.append(nn.BatchNorm1d(unflatten_shape_prod))
        self.fc.append(nn.ReLU())
        self.fc.append(nn.Unflatten(1, unflatten_shape))

        self.dec = nn.ModuleList()
        num_dec_kernels = [num_dec_kernels[0]] + num_dec_kernels
        num_enc_kernels = [num_enc_kernels[0]] + num_enc_kernels
        for i in range(len(num_dec_kernels)-1):
            input_size = num_enc_kernels[-i-1] + num_dec_kernels[i]
            output_size = num_dec_kernels[i+1]
            if i == 0:
                self.dec.append(DoubleConv(input_size, output_size))
            else:
                self.dec.append(UpBlock(input_size, output_size))
        self.dec.append(DoubleConv(num_dec_kernels[-1], image_size[0]))

        self.output_layers = nn.ModuleList()  # Fixed typo in variable name
        self.output_layers.append(nn.Conv2d(image_size[0], image_size[0],
                                            3, 1, 1))
        self.output_layers.append(nn.BatchNorm2d(image_size[0]))
        self.output_layers.append(nn.Sigmoid())

    def forward(self, x, enc_outputs):
        output = x
        for layer in self.fc:
            output = layer(output)
        for i, layer in enumerate(self.dec):
            output = torch.cat((output, enc_outputs[-i-1]), dim=1)
            output = layer(output)
        for layer in self.output_layers:
            output = layer(output)
        return output


if __name__ == "__main__":
    print("Testing Encoder_CVAE...")
    x = torch.zeros((10, 1, 256, 256))
    y = torch.zeros((10, 2))
    NUM_ENC_KERNELS = [19, 41, 97]
    NUM_DEC_KERNELS = [101, 53, 23]
    encoder = Encoder_CVUnet(image_size=x.shape[1:], n_z=26,
                             num_enc_kernels=NUM_ENC_KERNELS)
    decoder = Decoder_CVUnet(image_size=x.shape[1:], n_z=26,
                             num_enc_kernels=NUM_ENC_KERNELS,
                             num_dec_kernels=NUM_DEC_KERNELS)

    mu_logsigmasq, outputs = encoder.forward(x)
    for output in outputs:
        print(output.shape)
    print(mu_logsigmasq.shape)
