import torch
import torchvision


def load_data(args):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x : x*255)
    ])

    # Define the loader for training data.
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            '.', train=True, download=True,transform=transform), 
            batch_size=args.batch_size, 
            shuffle=True
    )

    # Define the loader for testing data.
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('.', train=False, download=True, transform=transform),
        batch_size=10, 
        shuffle=False
    )

    return train_loader, test_loader