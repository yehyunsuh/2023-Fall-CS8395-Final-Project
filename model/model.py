"""
Here goes the model
"""

import torch.nn as nn


class CVAE_MLP(nn.Module):
    def __init__(self):
        super(CVAE_MLP, self).__init__()


    def forward(self, x):
        
        return x
    

class CVAE_CNN(nn.Module):
    def __init__(self):
        super(CVAE_CNN, self).__init__()


    def forward(self, x):
        
        return x


def get_model(args):
    if args.model == "CVAE_MLP":
        model = CVAE_MLP()
    elif args.model == "CVAE_CNN":
        model = CVAE_MLP()

    return model