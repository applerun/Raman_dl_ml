import torch
import numpy


def pytorchlize(x: numpy.ndarray or torch.Tensor):
	for i in range(3 - len(x.shape)):
		if type(x) == numpy.ndarray:
			x = numpy.expand_dims(x, axis = len(x.shape) - 1)
		else:
			x = torch.unsqueeze(x, dim = len(x.shape) - 1)
	return x


def data2mean_std(spectrums: torch.Tensor or numpy.ndarray):
	"""
	:param spectrums:a batch of spectrums[b,l] or [b,1,l]
	:returns spectrum_mean ,spectrum_up=spectrum_mean+std,spectrum_down=spectrum_mean-std
	"""
	if type(spectrums) == numpy.ndarray:
		spectrums = torch.tensor(spectrums)
	if len(spectrums.shape) == 3 and spectrums.shape[1] == 1:
		spectrums = torch.squeeze(spectrums, dim = 1)
	elif len(spectrums.shape) > 2:
		raise AssertionError
	if len(spectrums.shape) == 1:
		return spectrums, spectrums, spectrums
	lenth = spectrums.shape[-1]
	batchsz = spectrums.shape[0]
	res_t = type(spectrums)
	y_mean = res_t(lenth)
	y_up = res_t(lenth)
	y_down = res_t(lenth)
	for idx in range(lenth):
		p = spectrums[:, idx]
		mean = p.mean()
		std = p.std()
		y_mean[idx] = mean
		y_up[idx] = mean + std
		y_down[idx] = mean - std
	return y_mean, y_up, y_down
