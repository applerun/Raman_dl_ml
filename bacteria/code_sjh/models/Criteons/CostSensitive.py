import numpy
import torch
from torch import nn
from bacteria.code_sjh.models.BasicModule import BasicModule, Flat
import torch.nn.functional as F
class CS_Criteon(BasicModule):
	def __init__(self,f,cost_matrix = None, num_classes = 2,):
		if cost_matrix is None:
			cost_matrix = torch.ones(num_classes, num_classes)
			for i in range(num_classes):
				cost_matrix[i, i] = 0

		assert cost_matrix.shape[0] == cost_matrix.shape[1] == num_classes,"cos_matrix:{},num_classes:{}".format(cost_matrix.shape,num_classes)
		super(CS_Criteon, self).__init__()
		self.Criteon = f

	def forward(self,output,label):#output:[b,n_c] label:[b]
		loss_t = self.Criteon(output,label)
