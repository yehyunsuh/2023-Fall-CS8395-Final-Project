import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader


class UnpairedDataset(Dataset):
    def __init__(self, args, df, transform=None, augmentation=None):
        self.args = args
        self.df = df.reset_index()
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.args.dataset_csv == "/data/yehyun/implantGAN/data/data.csv":
            image = plt.imread(f'/data/yehyun/implantGAN/{self.df.loc[index, "path"]}')
        else:
            image = cv2.imread(f'/data/yehyun/implantGAN/{self.df.loc[index, "path"]}')
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
            image_transformed = self.transform(image=image)['image']

        image_transformed = image_transformed.mean(axis=0, keepdims=True)
        if self.augmentation:
            image_augmented = self.augmentation(image=image)['image']
            image_augmented = image_augmented.mean(axis=0, keepdims=True)

            return image_transformed, image_augmented, label, patient_side
        else:
            return image_transformed, None, label, patient_side


class PairedDataset(Dataset):
    def __init__(self, args, df, transform=None, augmnentation=None):
        self.args = args
        df_pre = df[df['surgery'] == 'pre']
        df_pre = df_pre.rename(columns={'path': 'path_pre'})
        df_pre = df_pre.rename(columns={'filename': 'filename_pre'})
        df_pre = df_pre.drop(columns='surgery')
        df_pre = df_pre.drop_duplicates(subset=['patient', 'side'])

        df_post = df[df['surgery'] == 'post']
        df_post = df_post.rename(columns={'path': 'path_post'})
        df_post = df_post.rename(columns={'filename': 'filename_post'})
        df_post = df_post.drop(columns='surgery')
        df_post = df_post.drop_duplicates(subset=['patient', 'side'])

        self.df = pd.merge(left=df_pre, right=df_post)
        self.df = self.df.dropna()
        self.df = self.df.reset_index()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # image_pre = plt.imread(f'../{self.df.loc[index, "path_pre"]}')
        # image_post = plt.imread(f'../{self.df.loc[index, "path_post"]}')
        if self.args.dataset_csv == "/data/yehyun/implantGAN/data/data.csv":
            # image = plt.imread(f'/data/yehyun/implantGAN/{self.df.loc[index, "path"]}')
            image_pre = plt.imread(f'/data/yehyun/implantGAN/{self.df.loc[index, "path_pre"]}')
            image_post = plt.imread(f'/data/yehyun/implantGAN/{self.df.loc[index, "path_post"]}')
        else:
            # image = cv2.imread(f'/data/yehyun/implantGAN/{self.df.loc[index, "path"]}')
            image_pre = cv2.imread(f'/data/yehyun/implantGAN/{self.df.loc[index, "path_pre"]}')
            image_post = cv2.imread(f'/data/yehyun/implantGAN/{self.df.loc[index, "path_post"]}')

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
    if not args.augmentation:
        TRANSFORM = A.Compose([
            A.Resize(height=args.resize, width=args.resize),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ])
        AUGMENTATION = None
    else:
        TRANSFORM = A.Compose([
            A.Resize(height=args.resize, width=args.resize),
            # A.Rotate(always_apply=False, p=0.5, limit=(-15, 15), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False),
            A.ShiftScaleRotate(
                always_apply=False, p=0.5, 
                shift_limit_x=(-0.1, 0.1), shift_limit_y=(-0.1, 0.1), 
                scale_limit=(-0.15, 0.15), 
                rotate_limit=(-15, 15), interpolation=0, border_mode=0, value=(0, 0, 0), 
                mask_value=None, rotate_method='largest_box'
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ])
        AUGMENTATION = A.Compose([
            A.Resize(height=args.resize, width=args.resize),
            A.Rotate(always_apply=False, p=0.5, limit=(-15, 15), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False),
            # For dropout, compared image should be normal image
            A.OneOf([
                # A.PixelDropout(always_apply=True, p=1, dropout_prob=0.03, per_channel=0, drop_value=(0, 0, 0), mask_drop_value=None),
                A.CoarseDropout(always_apply=True, p=1, max_holes=20, max_height=20, max_width=20, min_height=15, min_width=15, fill_value=(0, 0, 0), mask_fill_value=None)
            ], p=0.5),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ])

    df = pd.read_csv(args.dataset_csv, encoding='utf-8')
    if args.dataset == "paired":
        train_dataset = PairedDataset(args, df, TRANSFORM, AUGMENTATION)
    elif args.dataset == "unpaired":
        train_dataset = UnpairedDataset(args, df, TRANSFORM, AUGMENTATION)

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
