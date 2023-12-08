import matplotlib.pyplot as plt
import numpy as np


def visualize_reconstructed_image(args, image, image_recon, epoch, idx, data_type, surgery_type, augmentation):
    if len(image) == 8:
        image = image.cpu().detach().numpy()
        image_recon = image_recon.cpu().detach().numpy()

        fig, ax = plt.subplots(4, 4) # 32 -> 24
        for i in range(4):
            for j in range(4):
                if j < 2:
                    ax[i, j].imshow(image[i*2+j, 0, :, :], cmap='gray')
                    ax[i, j].axis('off')
                else:
                    ax[i, j].imshow(image_recon[i*2+(j-2), 0, :, :], cmap='gray')
                    ax[i, j].axis('off')
        
        if data_type == "unpaired":
            if augmentation == "org":
                plt.savefig(f'{args.result}/{args.experiment_name}/Epoch{epoch}_Batch{idx}.png')
            else:
                plt.savefig(f'{args.result}/{args.experiment_name}/Epoch{epoch}_Batch{idx}_aug.png')
        elif data_type == "paired":
            plt.savefig(f'{args.result}/{args.experiment_name}/Epoch{epoch}_Batch{idx}_{surgery_type}.png')


    elif len(image) == 12:
        num = 12
        image = image[:num].cpu().detach().numpy()
        image_recon = image_recon[:num].cpu().detach().numpy()

        fig, ax = plt.subplots(4, 6) # 32 -> 24
        for i in range(4):
            for j in range(6):
                if j < 3:
                    ax[i, j].imshow(image[i*3+j, 0, :, :], cmap='gray')
                    ax[i, j].axis('off')
                else:
                    ax[i, j].imshow(image_recon[i*3+(j-3), 0, :, :], cmap='gray')
                    ax[i, j].axis('off')
        
        if data_type == "unpaired":
            if augmentation == "org":
                plt.savefig(f'{args.result}/{args.experiment_name}/Epoch{epoch}_Batch{idx}.png')
            else:
                plt.savefig(f'{args.result}/{args.experiment_name}/Epoch{epoch}_Batch{idx}_aug.png')
        elif data_type == "paired":
            plt.savefig(f'{args.result}/{args.experiment_name}/Epoch{epoch}_Batch{idx}_{surgery_type}.png')

    else:
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
            if augmentation == "org":
                plt.savefig(f'{args.result}/{args.experiment_name}/Epoch{epoch}_Batch{idx}.png')
            else:
                plt.savefig(f'{args.result}/{args.experiment_name}/Epoch{epoch}_Batch{idx}_aug.png')
        elif data_type == "paired":
            plt.savefig(f'{args.result}/{args.experiment_name}/Epoch{epoch}_Batch{idx}_{surgery_type}.png')
