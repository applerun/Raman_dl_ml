import torch
from torch import nn
from torch.nn import functional as F
import os, sys
from bacteria.code_sjh.models.BasicModule import BasicModule, Flat

__all__ = []


def makeBlk(
		Net
):
	class ResBlk(BasicModule):
		def __init__(
				self,
				ch_in,
				ch_out,
				stride = 1,
				**args,
		):
			super(ResBlk, self).__init__()
			self.net = Net(ch_in, ch_out, stride, **args)
			self.shortcut = nn.Sequential()  # shortcut层
			if ch_in != ch_out:
				self.shortcut = nn.Sequential(
					nn.Conv1d(ch_in, ch_out, kernel_size = 1, stride = stride, bias = False),
					nn.BatchNorm1d(ch_out)
				)

		def forward(
				self,
				x
		):
			return F.relu(self.net(x) + self.shortcut(x))

	return ResBlk


class ResNet(BasicModule):
	def __init__(
			self,
			sample_tensor: torch.Tensor,
			block,
			num_classes,
			planes = None,
			num_block = None, ):

		super(ResNet, self).__init__()

		if planes is None:
			planes = [64, 128, 256, 512]  # ch = 1 ->64 layer1
		if num_block is None:
			num_block = [2, 2, 2, 2]

		self.ch_in = sample_tensor.shape[-2]
		self.num_classes = num_classes

		# conv1
		self.conv1 = nn.Conv1d(self.ch_in, 64, kernel_size = 3, stride = 2, padding = 1)
		self.ch_in = 64
		self.bn1 = nn.BatchNorm1d(64)
		self.relu = nn.ReLU(inplace = True)
		self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)  # the first layer

		self.Conv1 = nn.Sequential(
			self.conv1,
			self.bn1,
			self.relu,
			self.maxpool
		)  # b,c,l ->b,c,l//4

		self.layer1 = self._make_layer(block, planes[0], num_block[0], stride = 1)  # four layers 2-5
		self.layer2 = self._make_layer(block, planes[1], num_block[1], stride = 2)  # four layers 6-9
		self.layer3 = self._make_layer(block, planes[2], num_block[2], stride = 2)  # four layers 10-13
		self.layer4 = self._make_layer(block, planes[3], num_block[3], stride = 2)  # four layers 14-17
		self.avg_pool = nn.AvgPool1d(kernel_size = 4)
		self.flat = Flat()

		self.features = nn.Sequential(
			self.Conv1,
			self.layer1,
			self.layer2,
			self.layer3,
			self.layer4,
		)
		with torch.no_grad():
			res = self.features(sample_tensor)  # [b,planes[-1],l]
			res = self.avg_pool(res)  # [b,planes[-1],l//4]
			res = self.flat(res)  # [b,planes[-1]*(l//4)]

		self.fc = nn.Sequential(
			nn.Linear(res.shape[-1], num_classes),
		)  # the last layer

	# conv2~5x
	def _make_layer(
			self,
			block,
			planes,
			num_blocks,
			stride
	):
		layers = []
		for i in range(num_blocks):
			if i == 0:
				layers.append(block(self.ch_in, planes, stride))
			else:
				layers.append(block(planes, planes, 1))

		self.ch_in = planes
		return nn.Sequential(*layers)



	def forward(
			self,
			x
	):

		x = self.features(x)
		x = self.avg_pool(x)
		x = self.flat(x)
		out = self.fc(x)
		return out


_resnet18 = dict(planes = [64, 128, 256, 512],
                 num_block = [2, 2, 2, 2], )
_resnet34 = dict(planes = [64, 128, 256, 512],
                 num_block = [3, 4, 6, 3], )
_resnet50 = dict(planes = [256, 512, 1024, 2048],
                 num_block = [3, 4, 6, 3], )
_resnet101 = dict(planes = [64, 128, 256, 512],
                  num_block = [3, 4, 23, 3], )
_resnet152 = dict(planes = [64, 128, 256, 512],
                  num_block = [3, 8, 36, 3], )


class net2(BasicModule):
	def __init__(
			self,
			ch_in,
			ch_out,
			stride,
			bias = False,
			padding = 1,
	):
		super(net2, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv1d(in_channels = ch_in, out_channels = ch_out, kernel_size = 3, stride = stride, padding = padding,
			          bias = bias),
			nn.BatchNorm1d(ch_out),
			nn.ReLU()
		)
		self.conv2 = nn.Sequential(
			nn.Conv1d(in_channels = ch_out, out_channels = ch_out, kernel_size = 3, stride = 1, padding = padding,
			          bias = bias),
			nn.BatchNorm1d(ch_out),
			nn.ReLU()
		)

	def forward(self,
	            x, ):
		x = self.conv1(x)
		x = self.conv2(x)
		return x


class net3(BasicModule):
	def __init__(self,
	             ch_in,
	             ch_out,
	             stride,
	             bias = False,
	             padding = 1,
	             ):
		super(net3, self).__init__()
		ch_m = ch_out // 4
		self.conv1 = nn.Sequential(
			nn.Conv1d(in_channels = ch_in, out_channels = ch_m, kernel_size = 1, stride = stride, padding = 0,
			          bias = bias),
			nn.BatchNorm1d(ch_m),
			nn.ReLU()
		)
		self.conv2 = nn.Sequential(
			nn.Conv1d(in_channels = ch_m, out_channels = ch_m, kernel_size = 3, stride = 1, padding = padding,
			          bias = bias),
			nn.BatchNorm1d(ch_m),
			nn.ReLU()
		)
		self.conv3 = nn.Sequential(
			nn.Conv1d(in_channels = ch_m, out_channels = ch_out, kernel_size = 1, stride = 1, padding = 0),
			nn.BatchNorm1d(ch_out),
			nn.ReLU()

		)

	def forward(self,
	            x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		return x


class ResNet18(ResNet):
	def __init__(self,
	             sample_tensor,
	             num_classes):
		block = makeBlk(Net = net2)
		super(ResNet18, self).__init__(
			sample_tensor,
			block,
			num_classes,
			**_resnet18,
		)
		self.model_name = "ResNet18"


class ResNet34(ResNet):
	def __init__(self,
	             sample_tensor,
	             num_classes):
		block = makeBlk(Net = net2)
		super(ResNet34, self).__init__(
			sample_tensor,
			block,
			num_classes,
			**_resnet34,
		)
		self.model_name = "ResNet34"


class ResNet50(ResNet):
	def __init__(self,
	             sample_tensor,
	             num_classes):
		block = makeBlk(Net = net3)
		super(ResNet50, self).__init__(
			sample_tensor,
			block,
			num_classes,
			**_resnet50,
		)
		self.model_name = "ResNet50"


class ResNet101(ResNet):
	def __init__(self,
	             sample_tensor,
	             num_classes):
		block = makeBlk(Net = net3)
		super(ResNet101, self).__init__(
			sample_tensor,
			block,
			num_classes,
			**_resnet101,
		)
		self.model_name = "ResNet101"


class ResNet152(ResNet):
	def __init__(self,
	             sample_tensor,
	             num_classes):
		block = makeBlk(Net = net3)
		super(ResNet152, self).__init__(
			sample_tensor,
			block,
			num_classes,
			**_resnet152,
		)
		self.model_name = "ResNet152"


if __name__ == '__main__':
	x = torch.Tensor(16, 1, 600)
	for Model in [ResNet18,ResNet34,ResNet50,ResNet101,ResNet152]:
		model = Model(x, 2)
		print(model)
		res = model(x)


