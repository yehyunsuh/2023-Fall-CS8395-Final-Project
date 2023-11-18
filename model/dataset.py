"""
Here goes the Data Loader
"""

import torch
import pandas as pd
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


class ImplantDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index()  ## reset index to get rid of the index for easier interpretation
        self.transform = transform


    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, index):
        image = None
        patient_info = None
        data_type = None

        if self.transform:
            pass

        return image, patient_info, data_type
    

def load_data(args):
    train_df = pd.read_csv(args.dataset_csv)

    # can add augmentations if you want
    transform = A.Compose([
        A.Resize(height=args.resize, width=args.resize),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])

    # Call Dataset class
    train_dataset = ImplantDataset(args, train_df, transform)
    print('Length of Dataset: ', train_dataset.__len__())

    # From class, call training data based on batch size
    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=num_workers)

    return train_loader