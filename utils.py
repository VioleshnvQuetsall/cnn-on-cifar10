import os
import sys

import time
import datetime

import yaml

from typing import Any

from torch.utils.tensorboard.writer import SummaryWriter


class Options:
    def __init__(self, options):
        _options = {
            k: Options(v) if isinstance(v, dict) else v
            for k, v in options.items()
        }

        def _to_dict():
            return {
                k: v[None] if isinstance(v, Options) else v
                for k, v in _options.items()
            }

        self.__dict__['_options'] = _options
        self.__dict__['_to_dict'] = _to_dict

    def __getattr__(self, key):
        return self.__dict__['_options'][key]

    def __setattr__(self, key, value):
        self.__dict__['_options'][key] = value

    def __getitem__(self, key):
        if key is None:
            return self.__dict__['_to_dict']()
        return self.__getattr__(key)

    __setitem__ = __setattr__


def parse_options(file):
    with open(file, mode='r') as f:
        options = yaml.safe_load(f)
    return Options(options)


class Logger:
    instances = []

    def __init__(self, log_dir, imple: Any = SummaryWriter):
        self.log_dir = log_dir
        self.enable = True
        if imple is sys.stdout:
            self.imple = imple
        elif isinstance(imple, str):
            self.imple = os.path.join(log_dir, imple)
        elif imple is SummaryWriter:
            self.imple = SummaryWriter(log_dir=log_dir)
        else:
            raise ValueError(f'unknown recognized logger object: {imple}')

        Logger.instances.append(self)

    def log(self, **kwargs):
        if not self.enable:
            return
        if self.imple is sys.stdout:
            print(*(f'[{k}: {v}]' for k, v in kwargs.items()), sep=' ')
        elif isinstance(self.imple, str):
            with open(self.imple, 'a') as f:
                print(*(f'[{k}: {v}]' for k, v in kwargs.items()),
                      sep=' ',
                      file=f)
        elif isinstance(self.imple, SummaryWriter):
            self.imple.add_scalar(kwargs['tag'], kwargs['value'], kwargs['step'])



def clear_dir(d, only_file):
    for file in os.listdir(d):
        path = os.path.join(d, file)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            clear_dir(path, only_file)
            if not only_file:
                os.rmdir(path)


def init_dir(base, name, suffix=None, dirs=None):
    if dirs is None:
        dirs = ['model', 'log', 'output']
    path = os.path.join(base, name)
    result = [path]
    if not os.path.isdir(path):
        os.mkdir(path)
    for dir_name in (f'{d}-{suffix}' if suffix else d for d in dirs):
        dir_path = os.path.join(path, dir_name)
        result.append(dir_path)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
    return result


class EmptyLoader:
    def __len__(self):
        return 0

    def __iter__(self):
        while True:
            yield None

    def __bool__(self):
        return False


class Timer:
    def __init__(self, current_time=None):
        self.reset(current_time)

    def reset(self, current_time=None):
        self.time = time.time() if current_time is None else current_time

    @property
    def record(self):
        return datetime.timedelta(seconds=time.time() - self.time)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            raise exc_value


if __name__ == '__main__':
    Logger(None, sys.stdout).log(test=None)
