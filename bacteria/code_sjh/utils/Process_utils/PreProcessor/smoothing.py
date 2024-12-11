from bacteria.code_sjh.utils.Process_utils.Core import *
from scipy.signal import savgol_filter
import numpy as np


def linearizeByterminal(slit: np.ndarray):
	shape = copy.deepcopy(slit.shape)
	l = len(slit.squeeze())
	for i in range(l):
		slit[i] = (l - i - 1) / (l - 1) * slit[0] + i / (l - 1) * slit[-1]
	return slit.reshape(shape)


class linearize_smoother(ProcessorFunction):
	def __init__(self, max_ = 1000,
				 span = 3, ):
		super(linearize_smoother, self).__init__("linearize_smoother(max_={},span={})".format(max_, span))
		self.max_ = max_
		self.span = span

	def __call__(self, y: np.ndarray, x = None):
		for i in range(len(y)):
			slit = y[max(0, i - self.span):min(len(y) - 1, i + self.span)]
			if y[i] - slit.min() > self.max_:
				y[max(0, i - self.span):min(len(y) - 1, i + self.span)] = linearizeByterminal(
					y[max(0, i - self.span):min(len(y) - 1, i + self.span)])
		return y if x is None else (y, x)


class sg_filter(ProcessorFunction):
	def __init__(self, window_length = 11,
				 poly_order = 3, ):
		super(sg_filter, self).__init__("sg_filter(window_length={},poly_order={})".format(window_length, poly_order))
		self.window_length = window_length
		self.poly_order = poly_order

	def __call__(self, y, x = None):
		y = savgol_filter(y, self.window_length, self.poly_order)
		if x is None:
			return y
		else:
			return y, x
