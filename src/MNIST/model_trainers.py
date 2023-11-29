import torch
import torch.nn as nn
import os

from loss import KL_div_N01
from visualization import visualize_reconstructed_image, visualize_loss_plot
from tqdm import tqdm


def train(args, model, train_loader, DEVICE):
    os.makedirs(f'results/{args.experiment_name}', exist_ok=True)

    # Define the loss function.
    loss_fn_MSE = nn.MSELoss()
    loss_fn_KL = KL_div_N01

    # Define the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    cvae_running_loss_list = []
    cvae_running_kl_loss_list = []
    cvae_running_mse_loss_list = []
    cvae_running_n_list = []
    for epoch in range(args.epochs):
        print(f"\nRunning Epoch #{epoch}")

        running_total_loss = 0.0
        running_mse_loss = 0.0
        running_kl_loss = 0.0
        running_n = 0

        for idx, (image, label) in enumerate(tqdm(train_loader)):
            labels_array = torch.zeros((len(label), 10))
            for i in range(len(labels_array)):
                labels_array[i][label[i]] = 1

            image = image.to(DEVICE)
            label = labels_array.to(DEVICE)
            image_recon, mean, log_sigma_sq = model.forward_train(image, label)

            mse_loss = loss_fn_MSE(image_recon, image)
            kl_loss = loss_fn_KL(mean, log_sigma_sq)
            loss = args.mse_weight * mse_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_total_loss += loss.item()
            running_mse_loss += mse_loss.item()
            running_kl_loss += kl_loss.item()
            running_n += image.shape[0]

        if epoch % int(args.epochs/10) == 0 or epoch == args.epochs-1:
            visualize_reconstructed_image(args, image, image_recon, epoch)
            print(
                f'loss: {running_total_loss / running_n:.6f}',
                f'KL loss: {running_kl_loss / running_n:.6f}',
                f'MSE loss: {running_mse_loss / running_n:.6f}')

        cvae_running_loss_list.append(running_total_loss/running_n)
        cvae_running_mse_loss_list.append(running_mse_loss/running_n)
        cvae_running_kl_loss_list.append(running_kl_loss/running_n)

    visualize_loss_plot(args, cvae_running_loss_list, cvae_running_mse_loss_list, cvae_running_kl_loss_list)
