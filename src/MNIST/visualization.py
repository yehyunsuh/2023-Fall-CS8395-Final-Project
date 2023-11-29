import matplotlib.pyplot as plt


def visualize_reconstructed_image(args, image, image_recon, epoch):
    image = image[:16].cpu().detach().numpy()
    image_recon = image_recon[:16].cpu().detach().numpy()

    # this part is hard coded to fit on batch size of 16
    fig, ax = plt.subplots(4, 8)
    for i in range(4):
        for j in range(8):
            if j < 4:
                ax[i, j].imshow(image[i*4+j, 0, :, :], cmap='gray')
                ax[i, j].axis('off')
            else:
                ax[i, j].imshow(image_recon[i*4+(j-4), 0, :, :], cmap='gray')
                ax[i, j].axis('off')
    
    plt.savefig(f'results/{args.experiment_name}/Epoch{epoch}.png')


def visualize_loss_plot(args, cvae_running_loss_list, cvae_running_mse_loss_list, cvae_running_kl_loss_list):
    fig, ax = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
    ax[0].set_title('Total Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].plot(cvae_running_loss_list, label='CVAE')
    ax[0].legend()

    ax[1].set_title('KL Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].plot(cvae_running_kl_loss_list, label='CVAE')
    ax[1].legend()

    ax[2].set_title('MSE Loss')
    ax[2].set_xlabel('Epoch')
    ax[2].plot(cvae_running_mse_loss_list, label='CVAE')
    ax[2].legend()

    plt.savefig(f'results/{args.experiment_name}/plot.png')