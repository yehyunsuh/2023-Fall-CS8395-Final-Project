import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import KL_div_N01


def train_unpaired(model, train_loader, device, lr, epochs):
    # define optimizer, loss function, and so on..
    loss_fn_MSE, loss_fn_KL = nn.MSELoss(), KL_div_N01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"\nRunning Epoch # {epoch}")

        running_total_loss = 0.
        running_mse_loss = 0.
        running_kl_loss = 0.

        for image, label, patient_side in tqdm(train_loader):
            image = image.to(device)

            image_recon, mean, log_sigma_sq = model.forward_train(image, label)

            mse_loss = loss_fn_MSE(image_recon, image)
            kl_loss = loss_fn_KL(mean, log_sigma_sq)
            loss = mse_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_total_loss += loss.item()
            running_mse_loss += mse_loss.item()
            running_kl_loss += kl_loss.item()
