import matplotlib.pyplot as plt
import numpy as np


def visualize_reconstructed_image(args, image, image_recon, epoch, idx, data_type, surgery_type):
    if len(image) < 16:
        num = 4
    elif len(image) >= 16:
        num = 16

    image = image[:num].cpu().detach().numpy()
    image_recon = image_recon[:num].cpu().detach().numpy()

    x = int(np.sqrt(num))
    y = int(np.sqrt(num) * 2)

    fig, ax = plt.subplots(x, y)
    for i in range(x):
        for j in range(y):
            if j < x:
                ax[i, j].imshow(image[i*x+j, 0, :, :], cmap='gray')
                ax[i, j].axis('off')
            else:
                ax[i, j].imshow(image_recon[i*x+(j-x), 0, :, :], cmap='gray')
                ax[i, j].axis('off')
    
    if data_type == "unpaired":
        plt.savefig(f'{args.result}/{args.experiment_name}/Epoch{epoch}_Batch{idx}.png')
    elif data_type == "paired":
        plt.savefig(f'{args.result}/{args.experiment_name}/Epoch{epoch}_Batch{idx}_{surgery_type}.png')
