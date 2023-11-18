"""
Here goes the main python script
Gary, you do not have to follow the outline here. Feel free to do it all on your taste.

I try to keep main.py as short as possible, so I usually create other .py files, 
but you can definitely just write everything here too.
"""

import torch
import argparse

from dataset import load_data
from model import get_model
from train import train
from utility import customize_seed


def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: set seed to fixed value
    customize_seed(8395)

    # TODO: Use WandB? Or any other logging tools?

    # TODO: Load Data
    train_loader = load_data(args)

    # TODO: Load Model
    model = get_model(args).to(DEVICE)

    # TODO: Train Model
    train(args, model, train_loader, DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # related to dataset
    parser.add_argument('--dataset_csv', type=str, default="", help='Path to the dataset csv file')
    parser.add_argument('--resize', type=int, default=256, help='Resize value')
    parser.add_argument('--batch_size', '--bs', type=int, default=8, help='Resize value')

    # related to model
    parser.add_argument('--model', type=str, default="CVAE_MLP", help='Name of the model that you want to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()

    main(args)