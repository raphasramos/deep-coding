""" File with the useful enums for the project """

import numpy as np
from enum import Enum, IntEnum
from functools import partial
import torch.nn as nn
import torch.optim as optim
from skimage.measure import compare_psnr, compare_ssim
from .msssim import MSSSIM

class Folders(IntEnum):
    """ Enum representing the names of the folders for output """
    TEST = 0
    VALIDATION = 1
    CHECKPOINTS = 2

    def __str__(self):
        return self.name.lower()


class ExecMode(IntEnum):
    """ Enum representing the possible modes for running the model """
    TRAIN = 0
    VALID = 1
    TEST = 2

    def __str__(self):
        return self.name.lower()


class Metrics(Enum):
    """ Enum representing the metrics used for comparison """
    PSNR = 0, partial(compare_psnr)
    SSIM = 1, partial(lambda x, y:
                      -10 * np.log10(1 - compare_ssim(x, y, multichannel=True)))
    MSSSIM = 2, partial(lambda x, y:
                        -10 * np.log10(1 - MSSSIM.compare_msssim(x, y)))

    def __new__(cls, value, func):
        member = object.__new__(cls)
        member._value_ = value
        member._func = func
        return member

    def __getitem__(self, key):
        if key == 0:
            return self.value
        if key == 1:
            return self._func.func
        return None

    def __str__(self):
        return self.name.lower()


class Codecs(IntEnum):
    """ Enum representing the codecs used in the code """
    NET = 0

    def __str__(self):
        return self.name.lower()


class OutputType(IntEnum):
    """ Enum representing the output considered in the code """
    NONE = 0
    RESIDUES = 1
    RECONSTRUCTION = 2

    def __str__(self):
        return self.name.lower()


class ImgData(Enum):
    """ Enum representing accepted ranges for pixels. """
    FLOAT = [0., 1.]
    UBYTE = [0, 255]

    def __str__(self):
        return self.name.lower()


class Optimizers(Enum):
    """ Enum represeting the acceptable optimizers """
    ADAM = partial(lambda lr, param: optim.Adam(param, lr=lr,
                                                weight_decay=1e-4, amsgrad=True))
    SGD = partial(lambda lr, param: optim.SGD(param, lr=lr, weight_decay=1e-4,
                                              momentum=0.95))

    def __str__(self):
        return self.name.lower()

    @classmethod
    def _missing_(cls, value):
        return cls.__members__[value.upper()]


class Losses(Enum):
    """ Enum with the available losses """
    MSE = nn.MSELoss()

    def __str__(self):
        return self.name.lower()

    @classmethod
    def _missing_(cls, value):
        return cls.__members__[value.upper()]
