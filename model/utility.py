import torch
import random
import numpy as np


def customize_seed(seed: int):
    """
    Functions that fixes the seed value so that we can reproduce the result with same hyperparameters
    :param seed (int): value of the seed
    :return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)