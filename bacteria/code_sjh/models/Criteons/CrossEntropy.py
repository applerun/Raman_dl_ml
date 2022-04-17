import numpy
import torch
from torch.nn import functional as F
from torch import nn

try:
    from ..BasicModule import BasicModule, Flat
except:
    from models.BasicModule import BasicModule, Flat
__all__ = ["BCE", "BCE_KLD"]


class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()


    def forward(self, x, y):

        return F.binary_cross_entropy(x, y, size_average = True)


class BCE_KLD(BasicModule):
    def __init__(self, beta):
        super(BCE_KLD, self).__init__()
        self.beta = beta

    def forward(self, x_hat, x, mu, sigma):
        bce = F.binary_cross_entropy(x, x_hat, size_average = False)
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) -
            torch.log(torch.pow(sigma, 2)) -
            1
        ) / numpy.prod(x.shape)
        return bce + self.beta * kld
