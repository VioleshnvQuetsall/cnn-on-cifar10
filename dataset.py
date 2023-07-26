import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as D
from torchvision.datasets import CIFAR10


def gcn(images, multiplier=55, eps=1e-10):
    # global contrast normalization
    images = images.astype(np.float64)
    images -= images.mean(axis=(1, 2, 3), keepdims=True)
    per_image_norm = np.sqrt(np.square(images).sum((1, 2, 3), keepdims=True))
    per_image_norm[per_image_norm < eps] = 1
    return multiplier * images / per_image_norm


def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, -1)
    image_cov = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
    zca_decomp = np.dot(U, np.dot(np.diag(1 / np.sqrt(S + eps)), U.T))
    mean = images.mean(axis=0)
    return mean, zca_decomp


def zca_normalization(images, mean, decomp):
    n_data, height, width, channels = images.shape
    images = images.reshape(n_data, -1)
    images = np.dot((images - mean), decomp)
    return images.reshape(n_data, height, width, channels)


class ImageDataset(D.Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.images, self.labels = dataset['images'], dataset['labels']
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


def cifar10_datasets(val_ratio=0.1, gcn_zca=False):
    train_dataset = CIFAR10('../assets', train=True, download=True)
    test_dataset = CIFAR10('../assets', train=False, download=True)

    if gcn_zca:
        train_dataset = {
            'images': gcn(train_dataset.data),
            'labels': torch.tensor(train_dataset.targets),
        }
        test_dataset = {
            'images': gcn(test_dataset.data),
            'labels': torch.tensor(test_dataset.targets),
        }

        mean, zca_decomp = get_zca_normalization_param(train_dataset['images'])
        train_dataset['images'] = zca_normalization(train_dataset['images'], mean,
                                                    zca_decomp).astype(np.float32)
        test_dataset['images'] = zca_normalization(test_dataset["images"], mean,
                                                zca_decomp).astype(np.float32)
    else:
        train_dataset = {
            'images': train_dataset.data,
            'labels': torch.tensor(train_dataset.targets),
        }
        test_dataset = {
            'images': test_dataset.data,
            'labels': torch.tensor(test_dataset.targets),
        }

    val_count = int(len(train_dataset['labels']) * val_ratio)
    val_dataset = {k: v[:val_count] for k, v in train_dataset.items()}
    train_dataset = {k: v[val_count:] for k, v in train_dataset.items()}

    return train_dataset, val_dataset, test_dataset
