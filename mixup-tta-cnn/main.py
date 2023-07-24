import os
import sys
from matplotlib import pyplot as plt

os.chdir(os.path.dirname(__file__))
sys.path.append('..')

import torch
import torch.optim as optim
import torch.utils.data as D
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

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
            optimizer,
            T_max=self.opt.epoch.n,
            last_epoch=self.opt.epoch.start,
            verbose=True)

        return [[optimizer, lr_scheduler]]

    def choose_optimizer_lr_scheduler(self, epoch):
        return 0

    def load_dataloaders(self, opt):

        transform_hflip = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(1),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_normal = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
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
        transform_tta = MultiTransformer([
            transform_crop_filp,
            transform_crop_filp,
            transform_crop_filp,
            transform_hflip,
            transform_normal,
        ])

        transform_train = T.Compose([
            T.ConvertImageDtype(torch.uint8),
            T.ToPILImage(),
            RandomSubtransform([transform_crop_filp, transform_filp])
        ])

        train_dataset, val_dataset, test_dataset = cifar10_datasets()
        train_dataset = ImageDataset(train_dataset, transform=T.ToTensor())
        val_dataset = ImageDataset(val_dataset, transform=transform_filp)
        test_dataset = ImageDataset(test_dataset, transform=transform_tta)

        # train_dataset = CIFAR10('../assets', train=True, transform=T.ToTensor())
        # val_dataset = CIFAR10('../assets', train=True, transform=transform_filp)
        # test_dataset = CIFAR10('../assets', train=False, transform=transform_tta)

        train_dataset = MultiDataset(
            [train_dataset, RandomDataset(train_dataset)],
            all_transform=RandomMixup(10, alpha=0.2, smoothing=0.05),
            transform=transform_train)

        train_loader = D.DataLoader(train_dataset,
                                    batch_size=opt.batch_size,
                                    shuffle=opt.shuffle)

        val_loader = D.DataLoader(val_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=opt.shuffle)

        test_loader = D.DataLoader(test_dataset,
                                   batch_size=opt.batch_size,
                                   shuffle=opt.shuffle,
                                   collate_fn=multi_collate)

        return train_loader, val_loader, test_loader

    def training_step(self, batch, batch_idx, optimizer):
        optimizer.zero_grad()
        x, y = (z.to(device) for z in batch)

        pred = self.model(x)
        loss = mixup_cross_entropy_loss(pred, y)

        loss.backward()
        optimizer.step()

        return {'Loss': loss.detach().cpu().numpy()}

    def predict(self, xs):
        # preds.shape == (Transfroms, BatchSize, ClassPrediction)
        preds = [self.model(x) for x in xs]
        weights = [2, 2, 2, 3, 3]
        assert len(preds) == len(weights)
        pred = sum(p * w / sum(weights) for p, w in zip(preds, weights))
        return pred

    def testing_step(self, batch, batch_idx):
        with torch.no_grad():
            xs, y = batch
            *xs, y = (z.to(device) for z in (*xs, y))

            pred = self.predict(xs)

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
            xs, y = batch
            *xs, y = (z.to(device) for z in (*xs, y))
            pred = self.predict(xs)
            p = pred.argmax(axis=1)
            self.stdout_logger.log(Acc=f'{accuracy_score(y, p):.4%}')
        return y, pred


def main():
    trainer = CNNTrainer('options.yaml')
    trainer.train()


if __name__ == '__main__':
    main()
