import numpy
import torch
import visdom
from torch import nn
# 记录各个路径
import os, sys, copy

coderoot = __file__
for i in range(3):
	coderoot = os.path.split(coderoot)[0]
checkpointroot = os.path.join(coderoot, "checkpoints")
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "data_AST")

sys.path.append(coderoot)

try:
	from ..BasicModule import BasicModule, Flat
	from ...utils import RamanData
	from bacteria.code_sjh.models.Parts import _CNN_encoder4, _TCNN_decoder4
except:
	from models.BasicModule import BasicModule, Flat
	from utils import RamanData
	from bacteria.code_sjh.models.Parts import _CNN_encoder4, _TCNN_decoder4

__all__ = ["DenseAutoEncoder", "ConvAutoEncoder", "muted_encoder", ]


class AutoEncoder(BasicModule):
	def save_encoder(self, filepath = None):
		if filepath is None:
			filepath = self.model_name + "_encoder.mdl"
		if not os.path.isabs(filepath):
			self.Encoder.save(os.path.join(checkpointroot, filepath))
		else:
			self.Encoder.save(filepath)

	def forward(self, x):
		shape = x.shape
		x = self.Flat(x)
		x = self.Encoder(x)
		x = self.Decoder(x)
		x = x.view(*shape)  # 形状还原
		return x

	def neck_vis(self, x, label, update = None, vis = None, markersymbol = 'cross', markersize = 6, win = 'neck vis'):
		assert self.neck_axis in [1, 2, 3], "too many neck axis"
		if not self.model_loaded:
			self.load(os.path.join(coderoot, "checkpoints", self.model_name + ".mdl"))
		if vis is None:
			vis = visdom.Visdom()
		with torch.no_grad():
			x = self.Flat(x)
			x = self.Encoder(x)
		if not type(label) == str:
			label = str(label)
		vis.scatter(x,
		            win = win,
		            update = update,
		            name = label,
		            opts = dict(
			            title = win,
			            markersize = markersize,
			            showlegend = True,
			            markersymbol = markersymbol,
		            )
		            )


class ConvAutoEncoder(AutoEncoder):
	def __init__(self, sample_tensor, neck_axis = 2):
		super(ConvAutoEncoder, self).__init__()
		self.neck_axis = neck_axis
		self.model_name = "convolutional_auto_encoder"
		self.Encoder = _CNN_encoder4(copy.deepcopy(sample_tensor), neck_axis = neck_axis)
		Til = self.Encoder.CNN_out_lenth
		self.Decoder = _TCNN_decoder4(sample_tensor, TCNN_in_lenth = Til, neck_axis = neck_axis)

	def forward(self, x):
		code = self.Encoder(x)
		x_hat = self.Decoder(code)
		return x_hat


class DenseAutoEncoder(AutoEncoder):
	def __init__(self, sample_tensor: torch.Tensor, neck_axis = 2, dropout = 0.1, bias = True):
		super(DenseAutoEncoder, self).__init__()
		# input = [b,c=1,l]
		self.neck_axis = neck_axis
		self.sample_tensor: torch.Tensor = sample_tensor.clone()
		self.model_name = "auto_encoder"
		self.dropout = dropout
		self.Flat = Flat()
		self.build_encoder()
		self.build_decoder(bias = bias)

	def build_encoder(self):
		with torch.no_grad():
			t = self.Flat(self.sample_tensor)
		self.lenth = t.shape[1]
		self.Encoder = nn.Sequential(
			nn.Linear(self.lenth, 256),
			nn.LeakyReLU(0.1),
			nn.Dropout(self.dropout),
			nn.Linear(256, 64),
			nn.LeakyReLU(0.1),
			nn.Dropout(self.dropout),
			nn.Linear(64, self.neck_axis),
			nn.LeakyReLU(0.1),

		)

	def build_decoder(self, bias = True):
		with torch.no_grad():
			t = self.Flat(self.sample_tensor)
		self.lenth = t.shape[1]
		self.Decoder = nn.Sequential(
			nn.Linear(self.neck_axis, 64, bias = bias),
			nn.LeakyReLU(0.1),
			nn.Dropout(self.dropout),
			nn.Linear(64, 256, bias = bias),
			nn.LeakyReLU(0.1),
			nn.Dropout(self.dropout),
			nn.Linear(256, self.lenth, bias = bias),
			nn.Sigmoid(),
		)


class muted_encoder():
	def __init__(self, module: BasicModule, mdlfile):
		# module.load(mdlfile)
		self.encoder = module.Encoder
		self.encoder.load(mdlfile)
		self.encoder.eval()
		return

	def __call__(self, input: numpy.ndarray):
		x = torch.tensor(input).to(torch.float32)
		x = RamanData.pytorchlize(x)

		out: torch.Tensor = self.encoder(x)
		out = torch.squeeze(out)

		out = out.detach().np()

		return out
