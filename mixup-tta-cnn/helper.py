import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as D


def accuracy_score(y, p):
    return (y == p).sum() / p.size(0)


class RandomMixup:
    def __init__(self, n_classes, alpha=0.2, smoothing=0.):
        self.n_classes = n_classes
        self.distribution = torch.distributions.beta.Beta(alpha, alpha)
        self.smoothing = smoothing

    def onehot(self, y):
        t = torch.full((self.n_classes, ),
                       self.smoothing / (self.n_classes - 1))
        t[y] = 1.0 - self.smoothing
        return t

    def __call__(self, images, labels):
        assert len(images) == len(labels) == 2
        r = self.distribution.sample()
        image = images[0] * r + images[1] * (1 - r)
        label = self.onehot(labels[0]) * r + self.onehot(labels[1]) * (1 - r)
        return image, label


def mixup_cross_entropy_loss(input, target, size_average=True):
    # input shape: (..., Batchsize, ClassPrediction)
    # target shape: (Batchsize, ClassPrediction)
    input = F.log_softmax(input, dim=-1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / target.size(0)
    return loss


class RandomDataset(D.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[np.random.randint(len(self))]


class MultiDataset(D.Dataset):
    def __init__(self,
                 datasets,
                 all_transform=None,
                 transform=None,
                 target_transform=None):
        assert datasets and all(
            [len(ds) == len(datasets[0]) for ds in datasets[1:]])
        self.datasets = datasets
        self.all_transform = all_transform
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        images, labels = zip(*(ds[index] for ds in self.datasets))
        if self.all_transform is not None:
            images, labels = self.all_transform(images, labels)
        if self.transform is not None:
            images = self.transform(images)
        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return images, labels


class MultiTransformer:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image) -> list:
        return [t(image) for t in self.transforms]


def multi_collate(batch):
    # batch: [[[image, ...], label], ...]
    images, labels = zip(*batch)
    # result: [(N, C, H, W), ...], labels
    return [D.default_collate(image_batch)
            for image_batch in zip(*images)], D.default_collate(labels)


def multi_dataloader(dataset, batch_size, shuffle):
    return D.DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=multi_collate)


class RandomSubtransform:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, img):
        return self.transforms[np.random.choice(len(self.transforms),
                                                p=self.p)](img)
