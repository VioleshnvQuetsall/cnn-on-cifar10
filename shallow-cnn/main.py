import os
import sys

os.chdir(os.path.dirname(__file__))
sys.path.append('..')

import numpy as np

import torch
from torch import nn
import torchvision.transforms as T
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.optim as optim

from trainer import Trainer
from model import CNN
from dataset import cifar10_datasets, ImageDataset
from helper import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNNTrainer(Trainer):
    def load_model(self, opt):
        return CNN(init_weights=True).to(device)

    def load_optimizers(self, opt):
        weight, bias = self.model.get_weight_bias()

        optimizer = optim.SGD([{
            'params': weight,
            'weight_decay': 1e-4,
            'initial_lr': opt.lr,
            'momentum': 0.9,
            'nesterov': True,
        }, {
            'params': bias,
            'weight_decay': 0,
            'initial_lr': opt.lr,
            'momentum': 0.9,
            'nesterov': True,
        }],
                              lr=opt.lr)

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.opt.epoch.n, last_epoch=self.opt.epoch.start,
            verbose=True)

        return [[optimizer, lr_scheduler]]

    def choose_optimizer_lr_scheduler(self, epoch):
        return 0

    def load_dataloaders(self, opt):
        transform_crop_filp = T.Compose([
            T.ToTensor(),
            T.RandomCrop(size=32, padding=4),
            T.RandomHorizontalFlip(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_filp = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_train = RandomSubtransform(
            [transform_crop_filp, transform_filp])

        transform_normal = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset, val_dataset, test_dataset = cifar10_datasets()
        train_dataset = ImageDataset(train_dataset, transform=transform_train)
        val_dataset = ImageDataset(val_dataset, transform=transform_normal)
        test_dataset = ImageDataset(test_dataset, transform=transform_normal)

        # train_dataset = CIFAR10('../assets', train=True, transform=transform_train)
        # val_dataset = CIFAR10('../assets', train=True, transform=transform_normal)
        # test_dataset = CIFAR10('../assets', train=False, transform=transform_normal)

        train_loader = DataLoader(train_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=opt.shuffle)
        test_loader = DataLoader(test_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=opt.shuffle)
        val_loader = DataLoader(val_dataset,
                                batch_size=opt.batch_size,
                                shuffle=opt.shuffle)

        return train_loader, val_loader, test_loader

    def training_step(self, batch, batch_idx, optimizer):
        optimizer.zero_grad()
        x, y = (z.to(device) for z in batch)
        pred = self.model(x)
        loss = F.cross_entropy(pred, y, label_smoothing=0.05)
        loss.backward()
        optimizer.step()

        accuracy = accuracy_score(y, pred.argmax(axis=1))
        # grad_norm = np.sqrt(sum(p.grad.detach().data.norm(2).item() ** 2
        #                         for p in self.model.parameters()
        #                         if p.grad is not None and p.requires_grad))

        return {
            'Loss': loss.detach().cpu().numpy(),
            'Accuracy': accuracy.detach().cpu().numpy()
        }

    def testing_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = (z.to(device) for z in batch)
            pred = self.model(x)
            loss = F.cross_entropy(pred, y, label_smoothing=0.05)
            accuracy = accuracy_score(y, pred.argmax(axis=1))

        return {
            'Loss': loss.detach().cpu().numpy(),
            'Accuracy': accuracy.detach().cpu().numpy()
        }

    def validating_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = (z.to(device) for z in batch)
            pred = self.model(x)
            loss = F.cross_entropy(pred, y, label_smoothing=0.05)
            accuracy = accuracy_score(y, pred.argmax(axis=1))

        return {
            'Loss': loss.detach().cpu().numpy(),
            'Accuracy': accuracy.detach().cpu().numpy()
        }

    def inference_step(self, batch):
        with torch.no_grad():
            x, y = (z.to(device) for z in batch)
            pred = self.model(x)
            p = pred.argmax(axis=1)
            self.stdout_logger.log(Acc=f'{accuracy_score(y, p):.4%}')
        return y, pred


def main():
    trainer = CNNTrainer('options.yaml')
    trainer.train()


if __name__ == '__main__':
    main()
