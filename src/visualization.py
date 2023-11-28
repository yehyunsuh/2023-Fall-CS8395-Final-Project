import matplotlib.pyplot as plt


def visualize_reconstructed_image(args, image, image_recon, epoch, idx, data_type, surgery_type):
    image = image.cpu().detach().numpy()
    image_recon = image_recon.cpu().detach().numpy()

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
    
    if data_type == "unpaired":
        plt.savefig(f'{args.result}/{args.experiment_name}/Epoch{epoch}_Batch{idx}.png')
    elif data_type == "paired":
        plt.savefig(f'{args.result}/{args.experiment_name}/Epoch{epoch}_Batch{idx}_{surgery_type}.png')
