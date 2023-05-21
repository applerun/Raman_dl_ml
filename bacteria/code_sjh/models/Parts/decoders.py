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
	from ..Parts.layers import _TCNN_layer
except:
	sys.path.append(coderoot)
	from models.BasicModule import BasicModule, Flat
	from models.CNN.Parts.layers import _CNN_layer

__all__ = ["_TCNN_decoder4","_dense_decoder"]

class _dense_decoder(BasicModule):
	def __init__(self, sample_tensor: torch.Tensor, neck_axis: int = 2, dropout: float = 0.1, bias: bool = True):
		lenth = sample_tensor.shape[-1]
		super(_dense_decoder, self).__init__(sample_tensor = sample_tensor)
		self.Decoder = nn.Sequential(
			nn.Linear(neck_axis, 64, bias = bias),
			nn.LeakyReLU(0.1),
			nn.Dropout(dropout),
			nn.Linear(64, 256, bias = bias),
			nn.LeakyReLU(0.1),
			nn.Dropout(dropout),
			nn.Linear(256, lenth, bias = bias),
			nn.Sigmoid(),
		)

	def forward(self, x):
		x = self.Encoder(x)
		return x


class _TCNN_decoder4(BasicModule):
	def __init__(self, sample_tensor, TCNN_in_lenth, neck_axis = 2, kernelsize = 8, stride = 3, dropout = 0.1,
	             verbose = False):
		super(_TCNN_decoder4, self).__init__(sample_tensor = sample_tensor)
		assert TCNN_in_lenth % 64 == 0, "CNN_in_lenth mod 64 must be 0"

		self.lenth = sample_tensor.shape[-1]
		self.channel = sample_tensor.shape[-2]
		self.TCNN_in_lenth = TCNN_in_lenth
		self.Dense = nn.Sequential(
			nn.Linear(neck_axis, 32),
			nn.ReLU(),
			nn.Dropout(dropout),

			nn.Linear(32, 128),
			nn.ReLU(),
			nn.Dropout(dropout),

			nn.Linear(128, TCNN_in_lenth),
			nn.ReLU(),
			nn.Dropout(dropout)

			# nn.Linear(neck_axis,TCNN_in_lenth)
		)  # [b,in_l]

		self.TCNN = nn.Sequential(
			# self, ch_in, ch_out, dropout = 0.1, kernelsize = 8, stride = 3, Maxpool = True, verbose = False
			_TCNN_layer(64, 64, dropout = dropout, kernel_size = kernelsize, stride = stride, verbose = verbose),
			# [b,64,l2]
			_TCNN_layer(64, 32, dropout = dropout, kernel_size = kernelsize, stride = stride, verbose = verbose),
			# [b,32,l3]
			_TCNN_layer(32, 32, dropout = dropout, kernel_size = kernelsize, stride = stride, verbose = verbose),
			# [b,32,l4]
			_TCNN_layer(32, 1, dropout = dropout, kernel_size = kernelsize, stride = stride, verbose = verbose),
			# [b,1,l5]
			nn.Sequential(
				nn.Conv1d(1, 1, kernel_size = kernelsize, stride = 1, ),
				nn.ReLU(),
				# nn.Sigmoid(),
				nn.Tanh(),
			)
		)

	def forward(self, x):
		out = self.Dense(x)  # [b,in_l]
		out = out.view(-1, 64, self.TCNN_in_lenth // 64)  # [b,64,in_l//64]
		out = self.TCNN(out)

		assert out.shape[1] == self.lenth, \
			" The length of the tensor({}) is not available, maybe u can reshape it to {}?" \
				.format(self.lenth, out.shape[-1])
		return out
