import argparse
import torch

from datasets import load_data
from models import get_model
from model_trainers import train
from utils import customize_seed


def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    customize_seed(8395)
    train_loader, _ = load_data(args)
    model = get_model(args).to(DEVICE)
    train(args, model, train_loader, DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # related to dataset
    parser.add_argument('--batch_size', '--bs', type=int, default=16, help='Resize value')

    # related to model
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--experiment_name', type=str, default="baseline", help='wandb name')
    parser.add_argument('--mse_weight', type=int, default=1)

    args = parser.parse_args()
    main(args)