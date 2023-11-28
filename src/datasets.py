import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class UnpairedDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index()
        self.transform = transform


    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        image = plt.imread(f'../{self.df.loc[index, "path"]}')
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

        image = image.mean(axis=0, keepdims=True)

        return image, label, patient_side


class PairedDataset(Dataset):
    def __init__(self, df, transform=None):
        df_pre = df[df['surgery'] == 'pre']
        df_pre = df_pre.rename(columns={'path': 'path_pre'})
        df_pre = df_pre.rename(columns={'filename': 'filename_pre'})
        df_pre = df_pre.drop(columns='surgery')

        df_post = df[df['surgery'] == 'post']
        df_post = df_post.rename(columns={'path': 'path_post'})
        df_post = df_post.rename(columns={'filename': 'filename_post'})
        df_post = df_post.drop(columns='surgery')

        self.df = pd.merge(left=df_pre, right=df_post)
        self.df = self.df.reset_index()
        self.transform = transform


    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        image_pre = plt.imread(self.df.loc[index, 'path_pre'])
        image_post = plt.imread(self.df.loc[index, 'path_post'])
        patient = self.df.loc[index, 'patient']
        side = self.df.loc[index, 'side']
        patient_side = patient + '_' + side
        label_pre = np.array([1, 0])
        label_post = np.array([0, 1])

        if self.transform:
            image_pre = self.transform(image=image_pre)['image']
            image_post = self.transform(image=image_post)['image']

        image_pre = image_pre.mean(axis=0, keepdims=True)
        image_post = image_post.mean(axis=0, keepdims=True)

        return image_pre, image_post, label_pre, label_post, patient_side


def load_data(args):
    TRANSFORM = A.Compose([
        # A.PadIfNeeded(),  # Apply padding
        A.Resize(height=args.resize, width=args.resize),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])
    df = pd.read_csv(args.dataset_csv)
    if args.dataset == "paired":
        train_dataset = PairedDataset(df, TRANSFORM)
    elif args.dataset == "unpaired":
        train_dataset = UnpairedDataset(df, TRANSFORM)
    
    print("Length of train dataset: ", len(train_dataset))
    return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


if __name__ == '__main__':
    df = pd.read_csv('data/data.csv')
    train_dataset = UnpairedDataset(df)
    print(len(train_dataset))
    sample = train_dataset[0]
    print(sample[0].shape)
    print(sample[1])
    print(sample[2])

    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=8, num_workers=num_workers)

    image, label, patient_side = next(iter(train_loader))
    print(image.shape)
    print(label.shape)
    print(patient_side)
    print('------------------')

    train_dataset = PairedDataset(df)
    print(len(train_dataset))
    sample = train_dataset[0]
    print(sample[0].shape)
    print(sample[1].shape)
    print(sample[2].shape)
    print(sample[3].shape)
    print(sample[4])

    num_workers = 4 * torch.cuda.device_count()
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=8, num_workers=num_workers)

    image_pre, image_post, label_pre, label_post, patient_side = next(iter(train_loader))
    print(image_pre.shape)
    print(image_post.shape)
    print(label_pre.shape)
    print(label_post.shape)
    print(patient_side)
    print('------------------')
