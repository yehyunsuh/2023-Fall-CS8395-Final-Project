import torch
import torch.nn as nn

from tqdm import tqdm


def train(args, model, train_loader, DEVICE):
    # define optimizer, loss function, and so on..
    loss_fn_MSE, loss_fn_KL = nn.MSELoss(), None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in args.epochs:
        print(f"\nRunning Epoch # {epoch}")

        total_loss = 0

        # TODO: train model
        for image, patient_info, data_type in tqdm(train_loader):
            image = image.to(DEVICE)

            prediction = model(image)

            # TODO: calculate loss & backward propagation + optimizer.zero_grad()
            loss = None

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO: Update total_loss or other variables for visualizing graphs
            total_loss += loss.item()
