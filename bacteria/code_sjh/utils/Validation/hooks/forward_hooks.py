import torch
import torch.nn as nn



class forward_hook():
	def __init__(self):
		self.fmap_block = []


	def __call__(self, module, data_input, data_output):
		self.fmap_block.append(data_output)



if __name__ == '__main__':
	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			self.conv1 = nn.Conv2d(1, 2, 3)
			self.pool1 = nn.MaxPool2d(2, 2)

		def forward(self, x):
			x = self.conv1(x)
			x = self.pool1(x)
			return x


	gc = forward_hook()
	net = Net()
	net.conv1.weight[0].fill_(1)
	net.conv1.weight[1].fill_(2)
	net.conv1.bias.data.zero_()
	net.conv1.register_forward_hook(gc)
	fake_img = torch.ones((1, 1, 4, 4))  # batch size * channel * H * W
	output = net(fake_img)

	print("output shape: {}\noutput value: {}\n".format(output.shape, output))
	print("feature maps shape: {}\nfeature maps value: {}\n".format(gc.fmap_block[0].shape, gc.fmap_block[0]))
