import torch
from torch import nn


def compute_gassian_kl(u1: torch.Tensor or float or int,
                       s1: torch.Tensor or float or int,
                       u2: torch.Tensor or float or int,
                       s2: torch.Tensor or float or int, ):
	KL = torch.log((s2 + 1e-8) / (s1 + 1e-8)) - 0.5 + 0.5 / pow(s2, 2) * (
			torch.pow(s1, 2) +
			torch.pow(u1 - u2, 2)
	)
	return KL


class Gaussian_KL(nn.Module):
	def __init__(self):
		super(Gaussian_KL, self).__init__()

	def forward(self, u1, s1, u2, s2):
		return compute_gassian_kl(u1, s1, u2, s2)
