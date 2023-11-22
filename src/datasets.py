"""
Here goes the Data Loader
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


TRANSFORM = A.Compose([
    # A.PadIfNeeded(),  # Apply padding
    A.Resize(height=256, width=256),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])


class UnpairedDataset(Dataset):
    def __init__(self, df, transform=TRANSFORM):
        self.df = df.reset_index()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = plt.imread(self.df.loc[index, 'path'])
        patient = self.df.loc[index, 'patient']
        side = self.df.loc[index, 'side']
        patient_side = patient + '_' + side
        surgery = self.df.loc[index, 'surgery']
        label = np.zeros((2,))
        if surgery == 'pre':
            label[0] = 1
        elif surgery == 'post':
            label[1] = 1
        if self.transform:
            image = self.transform(image=image)['image']

        return image, label, surgery, patient_side

# TODO: Implement paired dataset

# TODO: Implement data loader


def load_data(args):
    pass


if __name__ == '__main__':
    df = pd.read_csv('data/data.csv')
    train_dataset = UnpairedDataset(df)
    print(len(train_dataset))

    sample = train_dataset[0]
    print(sample[0].shape)
    print(sample[1])
    print(sample[2])
    print(sample[3])

    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=8, num_workers=num_workers)
    for i, (image, label, surgery, patient_side) in enumerate(train_loader):
        print(image.shape)
        print(label.shape)
        print(surgery)
        print(patient_side)
