import os
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

os.chdir(os.path.dirname(__file__))
sys.path.append('..')

from feature_visualize import feature_visualize, confusion_matrix
from utils import parse_options
from main import CNNTrainer


def visualize():
    opt = parse_options('options.yaml')
    opt.project.name = '3-layers'
    opt.epoch.start = 15
    trainer = CNNTrainer(opt)

    feature_visualize(trainer.model.conv, 64,
                      os.path.join(trainer.output_dir, '3.png'))


def cnn_confusion_matrix():
    opt = parse_options('options.yaml')
    opt.project.name = 'final'
    opt.epoch.start = 160
    opt.dataloader.batch_size = 128
    opt.dataloader.shuffle = False
    trainer = CNNTrainer(opt)

    confusion_matrix(trainer)


if __name__ == '__main__':
    # visualize()
    cnn_confusion_matrix()