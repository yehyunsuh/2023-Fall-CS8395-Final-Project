import torch
import torch.nn as nn
from tqdm import tqdm

from src.losses import KL_div_N01


def train_unpaired(model, train_loader, device, lr, epochs):
    loss_fn_MSE = nn.MSELoss()
    loss_fn_KL = KL_div_N01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"\nRunning Epoch #{epoch}")

        running_total_loss = 0.0
        running_mse_loss = 0.0
        running_kl_loss = 0.0

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


def train_paired(model, train_loader, device, lr, epochs):
    loss_fn_MSE = nn.MSELoss()
    loss_fn_KL = KL_div_N01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"\nRunning Epoch #{epoch}")

        running_total_loss = 0.0
        running_mse_loss = 0.0
        running_kl_loss = 0.0
        running_synth_loss = 0.0

        for image_pre, image_post, label_pre, label_post, patient_side in tqdm(train_loader):
            image_pre = image_pre.to(device)
            image_post = image_post.to(device)

            # pre-op reconstruction
            image_pre_recon, mean_pre, log_sigma_sq_pre = model.forward_train(image_pre, label_pre)
            mse_loss_pre = loss_fn_MSE(image_pre_recon, image_pre)
            kl_loss_pre = loss_fn_KL(mean_pre, log_sigma_sq_pre)

            # post-op reconstruction
            image_post_recon, mean_post, log_sigma_sq_post = model.forward_train(image_post, label_post)
            mse_loss_post = loss_fn_MSE(image_post_recon, image_post)
            kl_loss_post = loss_fn_KL(mean_post, log_sigma_sq_post)

            # synthesis post-op
            z_pre = model.encode_mean_logsigsq_to_z(mean_pre, log_sigma_sq_pre)
            image_post_synth = model.decode_z_to_output(z_pre, label_post)
            synth_loss = loss_fn_MSE(image_post_synth, image_post)

            # add losses
            loss = mse_loss_pre + kl_loss_pre + mse_loss_post + kl_loss_post + synth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_total_loss += loss.item()
            running_mse_loss += mse_loss_pre.item()
            running_kl_loss += kl_loss_pre.item()
            running_mse_loss += mse_loss_post.item()
            running_kl_loss += kl_loss_post.item()
            running_synth_loss += synth_loss.item()
