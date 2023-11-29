import torch
import argparse

from datasets import load_data
from models import get_model
from model_trainers import train
from utils import customize_seed
from log import initialize_wandb


def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: set seed to fixed value
    customize_seed(8395)

    # TODO: Use WandB? Or any other logging tools?
    if args.wandb:
        initialize_wandb(args)

    # TODO: Load Data
    train_loader = load_data(args)

    # TODO: Load Model
    model = get_model(args).to(DEVICE)

    # TODO: Train Model
    train(args, model, train_loader, DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # related to dataset
    parser.add_argument('--dataset', type=str, default="unpaired", help='To use unpaired or paired dataset')
    parser.add_argument('--dataset_csv', type=str, default="../data/data.csv", help='Path to the dataset csv file')
    parser.add_argument('--resize', type=int, default=256, help='Resize value')
    parser.add_argument('--batch_size', '--bs', type=int, default=16, help='Resize value')

    # related to model
    parser.add_argument('--model', type=str, default="CVAE_CNN", help='Name of the model that you want to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--mse_weight', type=int, default=1, help='Learning rate')

    # related to logging
    parser.add_argument('--wandb', action='store_true', help='whether to use wandb or not')
    parser.add_argument('--experiment_name', type=str, default="baseline", help='wandb name')
    parser.add_argument('--result', type=str, default="../result", help='path where results are saved')

    args = parser.parse_args()
    main(args)
