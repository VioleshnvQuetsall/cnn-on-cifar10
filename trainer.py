import time
import datetime
import sys
import random

from abc import ABC, abstractmethod
from collections import defaultdict
from os.path import join
from pprint import pprint

import numpy as np

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fix_random_seed(seed=42, cudnn=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if cudnn:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True


class Trainer(ABC):
    def __init__(self, options):
        opt = options if isinstance(options,
                                    Options) else parse_options(options)
        pprint(opt[None], sort_dicts=False)

        # opt
        self.opt = opt
        self.opt.epoch.n = self.adjust_epoch_n(opt.epoch)

        # dir
        dirs = self.init_dirs(self.opt.project)
        self.base_dir, self.model_dir, self.log_dir, self.output_dir = dirs
        if self.opt.epoch.start == 0 and self.opt.project.clear_log:
            clear_dir(self.log_dir, only_file=True)

        # model and optimizer
        self.model = self.load_model(self.opt.model)
        self.optimizers = self.load_optimizers(self.opt.optimizer)

        self.resume(self.opt.epoch.start)
        if self.opt.epoch.start != 0:
            self.opt.epoch.start += 1

        # dataloader
        loaders = [
            loader or EmptyLoader()
            for loader in self.load_dataloaders(self.opt.dataloader)
        ]
        self.train_loader, self.val_loader, self.test_loader = loaders

        # logger
        self.load_loggers(self.log_dir)

    def adjust_epoch_n(self, opt):
        return ((opt.n - 1) // opt.save_interval + 1) * opt.save_interval + 1

    def init_dirs(self, project):
        timestamp = project.timestamp
        timestamp = timestamp and datetime.datetime.now().strftime(
            str(timestamp))
        return init_dir(project.assets, project.name, timestamp)

    def load_loggers(self, log_dir):
        self.stdout_logger = Logger(None, sys.stdout)
        self.file_logger = Logger(log_dir, 'log.log')
        self.train_logger = Logger(join(log_dir, 'train'), SummaryWriter)
        self.test_logger = Logger(join(log_dir, 'test'), SummaryWriter)
        self.val_logger = Logger(join(log_dir, 'val'), SummaryWriter)

    @abstractmethod
    def load_model(self, opt):
        ...

    @abstractmethod
    def load_optimizers(self, opt):
        ...

    @abstractmethod
    def load_dataloaders(self, opt):
        ...

    @abstractmethod
    def training_step(self, batch, batch_idx, optimizer):
        ...

    @abstractmethod
    def testing_step(self, batch, batch_idx):
        ...

    @abstractmethod
    def validating_step(self, batch, batch_idx):
        ...

    @abstractmethod
    def choose_optimizer_lr_scheduler(self, epoch):
        ...

    def enumerate_loader_zip(self):
        val_max_iter = self.opt.dataloader.val_max_iter
        test_max_iter = self.opt.dataloader.test_max_iter

        val_max_iter = (len(self.val_loader) if val_max_iter is None else min(
            val_max_iter, len(self.val_loader)))
        test_max_iter = (len(self.test_loader) if test_max_iter is None else
                         min(test_max_iter, len(self.test_loader)))

        val_interval = len(self.train_loader) // val_max_iter + 1
        test_interval = len(self.train_loader) // test_max_iter + 1

        val_iterator = iter(self.val_loader)
        test_iterator = iter(self.test_loader)

        for i, train_batch in enumerate(self.train_loader):
            val_batch = next(val_iterator) if i % val_interval == 0 else None
            test_batch = next(
                test_iterator) if i % test_interval == 0 else None
            yield i, (train_batch, val_batch, test_batch)

    def testing(self, epoch):
        max_iter = self.opt.dataloader.test_max_iter
        max_iter = (len(self.test_loader) if max_iter is None else min(
            max_iter, len(self.test_loader)))

        self.model.eval()
        results = defaultdict(list)
        for i, batch in zip(range(max_iter), self.test_loader):
            # use len(train_loader) to keep x-axis aligned
            result = self.testing_step(
                batch, int((epoch + i / max_iter) * len(self.train_loader)))
            for k, v in result.items():
                results[k].append(v)
        return results

    def training_validating_testing(self, epoch):
        optimizer, lr_scheduler = self.optimizers[
            self.choose_optimizer_lr_scheduler(epoch)]

        results = [defaultdict(list) for _ in range(3)]
        for i, batches in self.enumerate_loader_zip():
            batch_idx = (epoch * len(self.train_loader) + i)
            step = batch_idx * self.opt.dataloader.batch_size
            train_batch, val_batch, test_batch = batches

            self.model.train()
            result = self.training_step(train_batch, batch_idx, optimizer)
            for k, v in result.items():
                results[0][k].append(v)
                self.train_logger.log(tag=k, value=v, step=step)

            self.model.eval()
            if val_batch is not None:
                result = self.validating_step(val_batch, batch_idx)
                for k, v in result.items():
                    results[1][k].append(v)
                    self.val_logger.log(tag=k, value=v, step=step)
            if test_batch is not None:
                result = self.testing_step(test_batch, batch_idx)
                for k, v in result.items():
                    results[2][k].append(v)
                    self.test_logger.log(tag=k, value=v, step=step)

        if lr_scheduler:
            lr_scheduler.step()
        return results

    def train(self):
        start, n = self.opt.epoch.start, self.opt.epoch.n
        with Timer() as t:
            for epoch in range(start, n):
                results = self.training_validating_testing(epoch)

                log_info = {'Epoch': f'{epoch:3d}/{n}'}
                for name, result in zip('train validate test'.split(),
                                        results):
                    for k, v in result.items():
                        log_info[f'{name} {k}'] = f'{np.mean(v):.4f}'
                log_info['ETA'] = t.record * (n - epoch - 1)

                t.reset()

                self.stdout_logger.log(**log_info)
                self.file_logger.log(**log_info)

                self.save(epoch)

    def save(self, epoch):
        if epoch != 0 and epoch % self.opt.epoch.save_interval == 0:
            self.model.eval()
            last_train = {
                'model':
                self.model.state_dict(),
                'optimizers':
                [[o.state_dict(), s.state_dict()] for o, s in self.optimizers]
            }
            torch.save(last_train, join(self.model_dir, f'{epoch}.pth'))

            print(f'save {epoch}.pth')

    def resume(self, start):
        if start != 0:
            last_train = torch.load(join(self.model_dir, f'{start}.pth'),
                                    map_location=torch.device(device))
            self.model.load_state_dict(last_train['model'])

            if self.opt.optimizer.resume_optimizers:
                for (o, s), (o_sd, s_sd) in zip(self.optimizers,
                                                last_train['optimizers']):
                    o.load_state_dict(o_sd)
                    s.load_state_dict(s_sd)

            print(f'resume {start}.pth')

    @abstractmethod
    def inference_step(self, batch):
        ...

    def inference(self, dataloader=None):
        self.model.eval()
        if dataloader is None:
            dataloader = self.test_loader
        ys, ps = [], []

        for batch in dataloader:
            y, pred = self.inference_step(batch)
            ys.append(y.numpy())
            ps.append(pred.numpy())
        y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
        return y_true, y_pred