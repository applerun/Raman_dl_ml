import os, sys
import torch
from torch import nn
import copy

coderoot = os.path.split(os.path.split(__file__)[0])[0]
coderoot = os.path.split(coderoot)[0]
checkpointroot = os.path.join(coderoot, "checkpoints")
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "data_AST")
sys.path.append(coderoot)

try:
	from bacteria.code_sjh.models.BasicModule import BasicModule, Flat
	from bacteria.code_sjh.utils import RamanData
except:
	from models.BasicModule import BasicModule, Flat
	from utils import RamanData


class _CNN_layer(BasicModule):
	def __init__(self, ch_in, ch_out, dropout = 0.1, kernelsize = 8, stride = 3, Maxpool = True, verbose = False):
		"""

		:param ch_in: 输入channel
		:param ch_out: 输出channel
		:param dropout: dropout层的值
		:param conv1d: 自定义卷积层
		"""

		super(_CNN_layer, self).__init__()
		self.model_name = "_CNN_layer"

		self.verbose = verbose

		if Maxpool:
			self.CNN = nn.Sequential(
				nn.Conv1d(ch_in, ch_out, (kernelsize), (stride), ),
				nn.BatchNorm1d(ch_out),
				nn.ReLU(),
				nn.MaxPool1d(kernel_size = (kernelsize), stride = 1, padding = kernelsize // 2),
				nn.Dropout(dropout)
			)
		else:
			self.CNN = nn.Sequential(
				nn.Conv1d(ch_in, ch_out, (kernelsize), (stride), ),
				nn.BatchNorm1d(ch_out),
				nn.ReLU(),
				nn.Dropout(dropout)
			)

	def forward(self, x):
		out = self.CNN(x)
		if self.verbose:
			print(self.model_name, ":", out.shape)
		return out


class _TCNN_layer(BasicModule):
	def __init__(self, ch_in, ch_out, dropout = 0.1, kernel_size = 8, stride = 3, verbose = False):
		"""

		:param ch_in: 输入channel
		:param ch_out: 输出channel
		:param dropout: dropout层的值
		:param conv1d: 自定义卷积层
		"""

		super(_TCNN_layer, self).__init__()
		self.model_name = "_TCNN_layer"
		self.CNN = nn.Sequential(
			nn.ConvTranspose1d(ch_in, ch_out, (kernel_size), (stride)),
			nn.BatchNorm1d(ch_out),
			nn.ReLU(),
			nn.Dropout(dropout)
		)
		self.verbose = verbose

	def forward(self, x):
		out = self.CNN(x)

		if self.verbose:
			print(self.model_name, ":", out.shape)
		return out


