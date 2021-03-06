import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import sys, os
import numpy
import copy

coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", )
sys.path.append(coderoot)
from bacteria.code_sjh.utils.Validation.visdom_utils import *


def spectrum_vis_mpl(spectrums: torch.Tensor,
                     xs = None,
                     name = None,
                     bias = 0,
                     side = True,
                     ax: plt.Axes = None,
                     line_color:str = None,
                     shadow_color:str = None,
                     ):
	"""

	:param spectrums: a batch of spectrums[b,l] or [b,1,l]
	:param xs: the x axis of the spectrum, default is None
	:param win: the window name in visdom localhost and the title of the window
	:param update: None if to rewrite and "append" if to add
	:param name: the name of the batch of spectrums, the line of the mean value is named as [name]+"_mean", mean + std
	-> [name] + "_up" , mean - std -> [name] + "_down"
	for example : If the name of the spectrums is "bacteria_R",then the names of the lines are "bacteria_R_mean",
	"bacteria_R_up" and "bacteria_R_down"
	:param opts: dict, {"main":dict(line visdom opts for the "mean" line), "side":dict(line visdom opts for the "up" line
	 and "down" line)}
	:param bias: move the lines of spectrums up(positive number) or down(negative number)
	:param side: whether to show the mean + std and mean - std line
	:returns spectrum_mean ,spectrum_up=spectrum_mean+std,spectrum_down=spectrum_mean-std
	"""

	if name is None:
		name = "spectrum"
	if shadow_color is None:
		shadow_color = "skyblue"
	if line_color is None:
		line_color = "blue"
	y_mean, y_up, y_down = data2mean_std(spectrums)
	y_mean += bias
	y_up += bias
	y_down += bias
	if xs is None:
		xs = torch.arange(spectrums.shape[-1])

	assert xs.shape[-1] == y_mean.shape[-1], r"lenth of xs and spectrums doesn't fit"

	fig: plt.Figure()
	if ax is None:
		fig, ax = plt.subplots(1, 1)
	else:
		fig = ax.figure

	ax.set_title(name)
	ax.set_xlabel("wavenumber cm^-^1")
	ax.set_ylabel("intensity")
	ax.plot(xs, y_mean, label = name,color = line_color)
	if side:
		ax.fill_between(xs, y_down, y_up,color= shadow_color)

	return fig


if __name__ == '__main__':
	from bacteria.code_sjh.utils.RamanData import Raman_dirwise, getRamanFromFile
	from scipy import interpolate
	readdatafunc0 = getRamanFromFile(wavelengthstart = 390, wavelengthend = 1810,
	                                 dataname2idx = {"Wavelength": 0, "Column": 2, "Intensity": 1}, )
	def readdatafunc(  # ?????????????????????????????????512
			filepath
	):
		R, X = readdatafunc0(filepath)
		R = numpy.squeeze(R)
		f = interpolate.interp1d(X, R, kind = "cubic")
		newX = numpy.linspace(400, 1800, 512)
		newR = f(newX)
		newR = numpy.expand_dims(newR, axis = 0)
		return newR, newX


	dir = os.path.join(projectroot, "data", "liver_cell")
	csvconfig_a = dict(dataroot = dir,
	                   LoadCsvFile = readdatafunc,
	                   backEnd = ".asc", )
	db = Raman_dirwise(**csvconfig_a, sfpath = "Raman_dirwise.csv", newfile = True, shuffle = False)
	label2data = db.get_data_sorted_by_label()
	label2name = db.label2name()
	fig, ax = plt.subplots(1, len(list(label2data.keys())), dpi = 200)
	for label in label2data.keys():
		data = label2data[label].numpy()
		name = label2name[label]
		spectrum_vis_mpl(data,db.xs,ax = ax[label],name = name)
	plt.subplots_adjust(wspace = 0.25)
	plt.show()

