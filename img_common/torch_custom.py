""" File containing custom operations and layers for torch models """

import torch
import torch.nn as nn
from torch.autograd import Function


class Noise(Function):
    def __init__(self):
        super(Noise).__init__()

    @staticmethod
    def forward(ctx, inpt, is_training=True):
        # Apply quantization noise while only training
        if is_training:
            prob = inpt.new(inpt.size()).uniform_()
            x = inpt.clone()
            x[(1 - inpt) / 2 <= prob] = 1
            x[(1 - inpt) / 2 > prob] = -1
            return x
        else:
            return inpt.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Sign(nn.Module):
    def __init__(self):
        super().__init__()
        self.Noise = Noise()

    def forward(self, x):
        x = self.Noise.apply(x, self.training)
        return x


class Binarizer(nn.Module):
    """ Torch Layer that implements binarization as in Toderici's article:
    Variable Rate Image Compression with Recurrent Neural Networks
    https://arxiv.org/abs/1511.06085.
    """
    def __init__(self):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, 32, kernel_size=1)
        self.sign = Sign()

    def forward(self, x):
        feat = self.conv(x)
        x = torch.tanh(feat)
        return self.sign(x)
