import os
import torch
import torch.nn as nn
from tqdm import tqdm

from losses import KL_div_N01
from log import log_unpaired_result, log_paired_result
from visualization import visualize_reconstructed_image


def train_unpaired(args, model, train_loader, device):
    loss_fn_MSE = nn.MSELoss()
    loss_fn_KL = KL_div_N01
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        print(f"\nRunning Epoch #{epoch}")

        running_total_loss = 0.0
        running_mse_loss = 0.0
        running_kl_loss = 0.0

        for idx, (image, label, patient_side) in enumerate(tqdm(train_loader)):
            image = image.to(device)
            label = label.to(torch.float32).to(device)

            image_recon, mean, log_sigma_sq = model(image, label)

            # visualize original and reconstructed image
            if (epoch % int(args.epochs/10) == 0 or epoch == args.epochs-1) and idx == 0 :
            # if (epoch % 100 == 0 or epoch == args.epochs-1) and idx == 0 :
                print(torch.max(image[0]), torch.min(image[0]))
                print(torch.max(image_recon[0]), torch.min(image_recon[0]))
                visualize_reconstructed_image(args, image, image_recon, epoch, idx, args.dataset, None)

            mse_loss = loss_fn_MSE(image_recon, image)
            kl_loss = loss_fn_KL(mean, log_sigma_sq)
            loss = args.mse_weight * mse_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_total_loss += loss.item()
            running_mse_loss += mse_loss.item()
            running_kl_loss += kl_loss.item()
        
        if args.wandb:
            log_unpaired_result(running_total_loss, running_mse_loss, running_kl_loss, len(train_loader))


def train_paired(args, model, train_loader, device):
    loss_fn_MSE = nn.MSELoss()
    loss_fn_KL = KL_div_N01
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        print(f"\nRunning Epoch #{epoch}")

        running_total_loss = 0.0
        running_mse_loss = 0.0
        running_kl_loss = 0.0
        running_synth_loss = 0.0

        for idx, (image_pre, image_post, label_pre, label_post, patient_side) in enumerate(tqdm(train_loader)):
            image_pre = image_pre.to(device)
            image_post = image_post.to(device)
            label_pre = label_pre.to(device)
            label_post = label_post.to(device)

            # pre-op reconstruction
            image_pre_recon, mean_pre, log_sigma_sq_pre = model.forward_train(image_pre, label_pre)
            mse_loss_pre = loss_fn_MSE(image_pre_recon, image_pre)
            kl_loss_pre = loss_fn_KL(mean_pre, log_sigma_sq_pre)

            # visualize original and reconstructed image
            if (epoch % int(args.epochs/10) == 0 or epoch == args.epochs-1) and idx == 0 :
                print(torch.max(image_pre[0]), torch.min(image_pre[0]))
                print(torch.max(image_pre_recon[0]), torch.min(image_pre_recon[0]))
                visualize_reconstructed_image(args, image_pre, image_pre_recon, epoch, idx, args.dataset, 'pre')

            # post-op reconstruction
            image_post_recon, mean_post, log_sigma_sq_post = model.forward_train(image_post, label_post)
            mse_loss_post = loss_fn_MSE(image_post_recon, image_post)
            kl_loss_post = loss_fn_KL(mean_post, log_sigma_sq_post)

            # visualize original and reconstructed image
            if (epoch % int(args.epochs/10) == 0 or epoch == args.epochs-1) and idx == 0 :
                print(torch.max(image_post[0]), torch.min(image_post[0]))
                print(torch.max(image_post_recon[0]), torch.min(image_post_recon[0]))
                visualize_reconstructed_image(args, image_post, image_post_recon, epoch, idx, args.dataset, 'post')

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
        
        if args.wandb:
            log_paired_result(running_total_loss, running_mse_loss, running_kl_loss, running_synth_loss, len(train_loader))


def train(args, model, train_loader, DEVICE):
    os.makedirs(f'{args.result}/{args.experiment_name}', exist_ok=True)
    if args.dataset == "paired":
        train_dataset = train_paired(args, model, train_loader, DEVICE)
    elif args.dataset == "unpaired":
        train_dataset = train_unpaired(args, model, train_loader, DEVICE)