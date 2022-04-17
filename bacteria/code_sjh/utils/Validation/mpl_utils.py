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

try:
	from .visdom_utils import data2mean_std
except:
	from visdom_utils import data2mean_std


def spectrum_vis_mpl(spectrums: torch.Tensor,
                     xs = None,
                     win = 'spectrum_vis_win_default',
                     update = None,
                     name = None,
                     opts: dict = None,
                     bias = 0,
                     side = True):
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
	if name == None:
		name = ""
	if opts is None:
		opts = dict(main = dict(
			xlabel = "cm-1",
		),
			side = dict(
				xlabel = "cm-1",
				dash = numpy.array(['dot']),
				linecolor = numpy.array([[123, 104, 238]]),
			))
	if not "main" in opts.keys() or not "side" in opts.keys():
		opts = dict(
			main = opts,
			side = opts
		)
	opts["main"].update(dict(title = win))
	opts["side"].update(dict(title = win))
	opts_down = copy.deepcopy(opts["side"])
	y_mean, y_up, y_down = data2mean_std(spectrums)
	if xs is None:
		xs = torch.arange(spectrums.shape[-1])

	assert xs.shape[-1] == y_mean.shape[-1], r"lenth of xs and spectrums doesn't fit"
	vis.line(X = xs, Y = y_mean + bias,
	         win = win,
	         name = name + "_mean",
	         update = update,

	         opts = opts["main"]
	         )
	if side:
		vis.line(X = xs, Y = y_up + bias,
		         win = win,
		         update = "append",
		         name = name + "_up",
		         opts = opts["side"], )

		vis.line(X = xs, Y = y_down + bias,
		         win = win,
		         update = "append",
		         name = name + "_down",
		         opts = opts_down, )

		return y_mean, y_up, y_down
