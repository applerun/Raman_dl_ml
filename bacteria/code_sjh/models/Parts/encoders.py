import copy
import torch
import visdom
import numpy
import os, sys
from torch import nn

coderoot = __file__
for i in range(4):
	coderoot = os.path.split(coderoot)[0]
checkpointroot = os.path.join(coderoot, "checkpoints")
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "data_AST")
try:
	from bacteria.code_sjh.models.BasicModule import BasicModule, Flat
	from ..Parts.layers import _CNN_layer
except:
	sys.path.append(coderoot)
	from models.BasicModule import BasicModule, Flat
	from models.Parts.layers import _CNN_layer

__all__ = ["_CNN_encoder4","_dense_encoder"]

class _CNN_encoder4(BasicModule):
	def __init__(self, sample_tensor, neck_axis = 2, dropout = 0.1, kernelsize = 8, verbose = False, stride = 3):
		super(_CNN_encoder4, self).__init__()
		self.CNN1 = nn.Sequential(
			# self, ch_in, ch_out, dropout = 0.1, kernelsize = 8, stride = 3, Maxpool = True, verbose = False
			_CNN_layer(1, 32, dropout = dropout, stride = stride, kernelsize = kernelsize, verbose = verbose),
			_CNN_layer(32, 32, dropout = dropout, stride = stride, kernelsize = kernelsize, verbose = verbose),
			_CNN_layer(32, 64, dropout = dropout, stride = stride, kernelsize = kernelsize, verbose = verbose),
			_CNN_layer(64, 64, dropout = dropout, stride = stride, kernelsize = kernelsize, Maxpool = False,
			           verbose = verbose),
			Flat(),
		)
		with torch.no_grad():
			sample_tensor1 = self.CNN1(sample_tensor)
			self.CNN_out_lenth = sample_tensor1.shape[-1]
		self.Dense1 = nn.Sequential(
			nn.Linear(self.CNN_out_lenth, 128),
			nn.ReLU(),
			nn.Dropout(dropout),
		)
		self.Dense2 = nn.Sequential(
			nn.Linear(128, 32),
			nn.ReLU(),
			nn.Dropout(dropout),
		)
		self.Dense3 = nn.Sequential(
			nn.Linear(32, neck_axis),
			nn.ReLU(),
			nn.Dropout(dropout),
		)

	def forward(self, x):
		out = self.CNN1(x)
		out = self.Dense1(out)
		out = self.Dense2(out)
		out = self.Dense3(out)
		return out  # [b,neck_axis]


class _dense_encoder(BasicModule):
	def __init__(self, sample_tensor: torch.Tensor, out_dim, dropout = 0.1):
		lenth = sample_tensor.shape[-1]
		super(_dense_encoder, self).__init__()
		self.Encoder = nn.Sequential(
			Flat(),
			nn.Linear(lenth, 256),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(256, 64),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(64, out_dim),
			nn.ReLU(),
		)

	def forward(self, x):
		x = self.Encoder(x)
		return x
