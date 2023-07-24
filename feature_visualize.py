import os
import sys

os.chdir(os.path.dirname(__file__))
sys.path.append('..')

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid

from utils import Timer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeNormalize:
    def __init__(self, mean, std):
        mean = torch.tensor(mean).view((1, -1, 1, 1))
        std = torch.tensor(std).view((1, -1, 1, 1))

        self.mean = mean
        self.std = std

    def __call__(self, x, clamp=False):
        x = (x * self.std) + self.mean
        if clamp:
            x.clamp_(0, 1)
        return x


normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
denormalize = DeNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


class FeatureVisualizer():
    def __init__(self, model: nn.Module):
        self.model = model.eval().to(device)

    def visualize(self, input_channel, size, channels,
                  upscaling_steps=12, upscaling_factor=1.2,
                  lr=1, steps=50, blur=None):
        sz = size
        img = torch.normal(mean=0.5, std=0.5,
                           size=(input_channel, sz, sz)).to(device)

        for i in range(upscaling_steps):
            img = normalize(img).unsqueeze(0)
            img = Variable(img, requires_grad=True)
            optimizer = torch.optim.Adam([img], lr=lr, weight_decay=1e-6)
            losses = []

            for j in range(steps):
                optimizer.zero_grad()
                output = self.model(img)
                loss = -output[0, channels].mean()
                loss.backward()

                losses.append(loss.item())
                print(f'[Epoch: {i+1:2d}/{upscaling_steps}] '
                      f'[Step: {j+1:3d}/{steps}] '
                      f'[total loss: {losses[-1]:.4f}] ')

                optimizer.step()

            print(f'[{i+1}/{upscaling_steps}: {np.mean(losses):.2f}]')
            with torch.no_grad():
                img = denormalize(img).detach().squeeze(0)
            sz = int(upscaling_factor * sz)
            img = T.Resize(sz, T.InterpolationMode.BICUBIC,
                           antialias=True)(img)
            if blur:
                img = T.GaussianBlur(kernel_size=blur)(img)
        return img


def save_images(images, nrow, path):
    image = make_grid(images, nrow=nrow, normalize=True)
    save_image(image, path)
    print(f'save {path}')


def feature_visualize(model, channel_count, path, channel=3, size=32,
                      upscaling_steps=12, upscaling_factor=1.2,
                      lr=0.1, steps=80, blur=None):
    visualizer = FeatureVisualizer(model)

    images = []
    with Timer() as t:
        for c in range(channel_count):
            print(c)
            img = visualizer.visualize(channel, size, [c],
                                       upscaling_steps=upscaling_steps,
                                       upscaling_factor=upscaling_factor,
                                       lr=lr, steps=steps, blur=blur)

            print(f'[channel: {c}/{channel_count}] '
                  f'[ETA: {t.record * (channel_count - c)}]')
            t.reset()

            images.append(img)

    save_images(images, int(np.sqrt(channel_count)), path)

def confusion_matrix(trainer, name='confusion matrix.png'):
    result = trainer.inference()
    np.save('b', result[1])
    result = result[0], result[1].argmax(axis=1)
    ConfusionMatrixDisplay.from_predictions(
        *result,
        display_labels=
        'airplane automobile bird cat deer dog frog horse ship truck'.split(),
        xticks_rotation='vertical')
    print(accuracy_score(*result))
    plt.show()
    plt.savefig(os.path.join(trainer.output_dir, name))


if __name__ == '__main__':
    feature_visualize()
