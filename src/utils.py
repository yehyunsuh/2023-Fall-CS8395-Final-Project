import torch
import random
import numpy as np


def customize_seed(seed: int):
    """
    Functions that fixes the seed value so that we can reproduce the result
    with same hyperparameters
    :param seed (int): value of the seed
    :return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def convert_image_to_scalar(img: torch.Tensor) -> torch.Tensor:
    """For the MLP version, we need to convert the image to scalar
    before feeding it to the model

    Args:
        img (torch.Tensor): the input image; shape: (B, C, H, W)

    Returns:
        torch.Tensor: converted scalar; shape: (B, C*H*W)
    """
    return img.view(img.shape[0], -1).clone().detach()


def convert_scalar_to_image(scalar: torch.Tensor,
                            shape: tuple = (10, 1, 28, 28)) -> torch.Tensor:
    """For the MLP version, we need to convert the output scalar to image

    Args:
        scalar (torch.Tensor): scalar shape: (B, C*H*W)
        shape (tuple, optional): target shape. Defaults to (10, 1, 28, 28).

    Returns:
        torch.Tensor: converted image; shape: (B, C, H, W)
    """
    return scalar.view(*shape).clone().detach()


if __name__ == "__main__":
    img = torch.rand((10, 1, 32, 32))
    print(img.shape)
    flat = convert_image_to_scalar(img)
    print(flat.shape)
    recon = convert_scalar_to_image(flat, (10, 1, 32, 32))
    print(recon.shape)
