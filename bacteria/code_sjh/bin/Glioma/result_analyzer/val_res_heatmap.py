import csv
import os

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy
import numpy as np
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap


def create_colormap_lightness(color: tuple,
							  sample = 512,
							  white = (1, 1, 1)):
	"""


	:param color:最大值颜色
	:param sample: 颜色采样数量，default=512
	:param white: 设置最大亮度的颜色，默认白色，default = (1, 1, 1)
	:return:ListedColormap：从白色到指定颜色的Colormap（lightness从大到小）
	"""
	xs = np.linspace(0, 1, sample)
	color_ = np.array(color)
	res = np.outer(np.ones(sample), np.array(white))
	np.ones((sample, 3))
	res += np.outer(xs, (color_ - 1))
	return ListedColormap(res)


def create_colormap_bluered(red = (1, 0, 0),
							blue = (0, 0, 1),
							sample_red = 256,
							sample_blue = 256,
							white = (1, 1, 1)):
	sample = sample_red
	xs = np.linspace(0, 1, sample)
	color_red = np.array(red)
	res_red = np.outer(np.ones(sample), np.array(white))
	res_red += np.outer(xs, (color_red - white))
	sample = sample_blue
	color_blue = np.array(blue)
	res_blue = np.outer(np.ones(sample), np.array(white))
	np.ones((sample, 3))
	res_blue += np.outer(xs, (color_blue - white))
	res_blue = res_blue[::-1, ]
	res = np.vstack((res_blue, res_red))
	return ListedColormap(res)


def legend_show(ax,
				color,
				hatches,
				hatch_color,
				pos = (0, 0),
				width = 0.8,
				height = 0.3,
				interval_y = 0.2):
	len_data = len(hatches)
	for i in range(len_data):
		hatch = hatches[i]
		ax.bar(x = pos[0], height = height, bottom = pos[1] + i * (height + interval_y), width = width, color = color,
			   hatch = hatch,
			   edgecolor = hatch_color)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.axis("off")
	ax.set_xticks([])
	ax.set_yticks([])


def plotdata(HeatData1,
			 HeatData2,
			 ax: matplotlib.axes.Axes,
			 cmap = "Reds",
			 hatch_color: str or tuple = "orange",
			 hatchs = None,
			 norm = colors.Normalize(0.5, 1, clip = True),
			 width = 0.8,
			 height = 0.5,
			 interval_x = 0,
			 interval_1 = 0.2,
			 interval_2 = 0.6,
			 interval_y = 0.05,
			 colorbar_orientation = "vertical",
			 count_1 = 3,
			 ticksize = 20
			 ):
	if hatchs is None:
		hatchs = ['//', '..', r"\\", '|', '-', '+', 'x', 'o', 'O', '.', '*']
	lenx = HeatData2.shape[1]

	if type(cmap) == list:
		cmaps = cmap
	else:
		cmaps = [cmap]

	for x_index in range(2 * lenx):
		mapper = cm.ScalarMappable(norm = norm, cmap = cmaps[x_index % (len(cmaps))])
		data_s = HeatData1[:, x_index] if x_index < lenx else HeatData2[:, x_index - lenx]
		x_pos = (width + interval_x) * x_index \
				+ x_index // count_1 * interval_1 \
				+ x_index // lenx * interval_2 \
			# + x_index // (2 * count_1) * interval_1
		hatch = hatchs[x_index % 3]
		plot_s(ax, data_s, hatch, x_pos, width, height, mapper = mapper, hatch_color = hatch_color,
			   interval_y = interval_y)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.axis("off")
	ax.set_xticks([])
	ax.set_yticks([])

	if len(cmaps) == 1:
		colorbars(cmaps, ax, norm = norm, colorbar_orientation = colorbar_orientation, ticksize = ticksize)
	fig = ax.figure
	return fig


def colorbars(cmaps,
			  plot_ax:matplotlib.axes.Axes = None,
			  norm = colors.Normalize(0.5, 1, clip = True),
			  colorbar_orientation = "vertical",
			  ticksize = 20):
	if plot_ax is None:
		try:
			fig_, axes = plt.subplots(1, len(cmaps), width_ratios = [0.1] * len(cmaps), squeeze = False)
		except AttributeError:
			fig_, axes = plt.subplots(1, len(cmaps), gridspec_kw = dict(width_ratios = [0.1] * len(cmaps)),
									  squeeze = False)
		axes = axes.flatten()
		for ax in axes:
			ax.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)
			ax.spines['bottom'].set_visible(False)
			ax.spines['left'].set_visible(False)
			ax.axis("off")
			ax.set_xticks([])
			ax.set_yticks([])
	else:
		fig_ = plot_ax.figure
	for i in range(len(cmaps) - 1, -1, -1):
		c = cmaps[i]
		mapper = cm.ScalarMappable(norm = norm, cmap = c)
		cb = fig_.colorbar(mapper, ax = axes[i], orientation = colorbar_orientation) if plot_ax is None \
			else fig_.colorbar(mapper, ax = plot_ax, orientation = colorbar_orientation)
		cb.ax.tick_params(labelsize = ticksize)
	return fig_


def plot_s(ax: matplotlib.axes.Axes,
		   data,
		   hatch,
		   pos,
		   width,
		   height,
		   mapper: cm.ScalarMappable = None,
		   hatch_color = "black",
		   interval_y = 0,
		   color = None):
	assert mapper is not None or color is not None
	for i in range(len(data)):
		value = data[-1 - i]
		color_ = mapper.to_rgba(value) if color is None else color
		# color = plt.cm.viridis(norm(value))
		ax.bar(x = pos, height = height, bottom = i * (height + interval_y), width = width, color = color_,
			   hatch = hatch,
			   edgecolor = hatch_color)
		ax.bar(x = pos, height = height, bottom = i * (height + interval_y), width = width, color = (0, 0, 0, 0),
			   edgecolor = color_)
	return


def main_hatchwise(HeatData1 = None,
				   HeatData2 = None,
				   dst = None,
				   skiprows = 1,
				   norm = colors.Normalize(0.5, 1, clip = True)
				   ):  # 纹理热力图
	if dst is not None and not os.path.isdir(dst):
		os.makedirs(dst)
	if HeatData1 is None:
		HeatData1 = np.loadtxt("Glioma_heatmap_python/plot_res/heatdata1.csv", delimiter = ",", skiprows = 1)
	if HeatData2 is None:
		HeatData2 = np.loadtxt("Glioma_heatmap_python/plot_res/heatdata2.csv", delimiter = ",", skiprows = 1)
	if os.path.isfile(HeatData1):
		HeatData1 = np.loadtxt(HeatData1, delimiter = ",", skiprows = skiprows)
	if os.path.isfile(HeatData2):
		HeatData2 = np.loadtxt(HeatData2, delimiter = ",", skiprows = skiprows)
	plt.rcParams['figure.figsize'] = (24, 9)

	hatches = ['//', '..', r"\\"]
	hatch_color = (1 / 1.5, 165 / 255 / 1.5, 0, 0)
	cmap = create_colormap_lightness((1, 0, 0))
	# cmap = cmap.reversed()
	cmap = create_colormap_bluered(white = (0.98, 0.98, 0.97))

	wh12y = np.array([2, 0.8, 0, 0.2, 1, 0.05])
	wh12y *= 2

	HeatData1_dl = HeatData1[:, 1::3]

	HeatData1_ml = np.delete(HeatData1, list(range(1, HeatData1.shape[1], 3)), axis = 1)

	HeatData2_dl = HeatData2[:, 1::3]
	HeatData2_ml = np.delete(HeatData2, list(range(1, HeatData2.shape[1], 3)), axis = 1)
	fig, ax = plt.subplots(figsize = (12, 9))
	plotdata(HeatData1_dl, HeatData2_dl, ax, cmap, hatch_color, hatches, norm, *wh12y,
			 colorbar_orientation = "vertical", ticksize = 25, count_1 = 1)
	# plt.savefig("fig.png")
	# fig, ax = plt.subplots()
	# legend_show(ax, "red", hatches, "orange",pos = (0,0.8*10))
	legend_show(ax, (0, 0, 0, 0), hatches[::-1], hatch_color,
				pos = (0, 0.8 * 10), interval_y = 0.05)
	plt.savefig("legend_dl.png" if dst is None else os.path.join(dst, "legend_dl.png"))
	colorbars([cmap], norm = norm)
	plt.savefig("legend_colorbar_dl.png" if dst is None else os.path.join(dst, "legend_colorbar_dl.png"))
	fig, ax = plt.subplots(figsize = (24, 9))
	plotdata(HeatData1_ml, HeatData2_ml, ax, cmap, hatch_color, hatches, norm, *wh12y,
			 colorbar_orientation = "vertical", ticksize = 25, count_1 = 2)
	plt.savefig("legend_ml.png" if dst is None else os.path.join(dst, "legend_ml.png"))
	fig, ax = plt.subplots(figsize = (36, 9))
	plotdata(HeatData1, HeatData2, ax, cmap, hatch_color, hatches, norm, *wh12y,
			 colorbar_orientation = "vertical", ticksize = 25, count_1 = 3)
	plt.savefig("legend_all.png" if dst is None else os.path.join(dst, "legend_all.png"))
	print("data saved in:{}".format(dst))


def main_colorwise():  # 颜色区分热力图
	plt.rcParams['figure.figsize'] = (28, 10)
	fig, ax = plt.subplots()

	cmap = [
		create_colormap_lightness((0.1, 0.4, 1)),
		create_colormap_lightness((0.2, 0.2, 1)),
		create_colormap_lightness((0.4, 0.1, 1))
	]
	plotdata(ax, hatch_color = (0, 0, 0, 0), cmap = cmap)
	plt.savefig("legend_color.png")
	# fig, ax = plt.subplots()
	fig = colorbars(cmap)
	fig.savefig("colorbars.png")


def getheatdata(dir,
				prefix,
				backend = ".csv",
				skiprows = 2,
				skipcolumns = 1,
				):
	confm2model2digit = None
	for stat_file in os.listdir(dir):
		if not (stat_file.startswith(prefix) and stat_file.endswith(backend)):
			continue
		stat_file_abs = os.path.join(dir, stat_file)
		datas = []
		with open(stat_file_abs) as f:
			reader = csv.reader(f).__iter__()
			headers = []
			for r in range(skiprows):
				headers.append(reader.__next__()[skipcolumns:])
			for data in reader:
				datas.append(data[skipcolumns:])
		datas = numpy.array(datas).T
		confms = headers[0]
		if confm2model2digit is None:
			confm2model2digit = {confm: {} for confm in confms}
		for x, confm in enumerate(confms):
			model = headers[1][x]
			digit = datas[x]
			confm2model2digit[confm][model] = digit

	return confm2model2digit


def merge_stat_files(src_dir,
					 dst_f,
					 prefix,
					 backend = ".csv",
					 skiprows = 2,
					 skipcolumns = 1,
					 new_models = None,
					 new_rows = 1):
	confm2model2digit = getheatdata(src_dir, prefix, backend, skiprows, skipcolumns)
	confms = list(confm2model2digit)
	if new_models is None:
		new_models = list(list(confm2model2digit.items())[0])
	res = [None] * (len(confms) * len(new_models))
	for x1, confm in enumerate(confms):
		for x2, model in enumerate(new_models):
			digit = list(confm2model2digit[confm][model])
			res[x1 * len(new_models) + x2] = [confm, model][:new_rows] + digit
	res = numpy.array(res).T
	numpy.savetxt(dst_f, res, fmt = "%s", delimiter = ",")


def main():
	from bacteria.code_sjh.Core.basic_functions.path_func import getRootPath

	projectroot = getRootPath("Raman_dl_ml")
	res_Heatmap_root = os.path.join(projectroot, "results", "glioma", "Heatmap", "20240318")
	l_prefix = "res_stat-data_all,res_stat-data_GBM,res_stat-data_batch123".split(",")

	for res_stat_dir in os.listdir(res_Heatmap_root):
		res_stat_dir = os.path.join(res_Heatmap_root, res_stat_dir)
		if not os.path.isdir(res_stat_dir): continue
		for prefix in l_prefix:
			merge_stat_files(res_stat_dir, os.path.join(res_stat_dir + prefix + ".csv"), prefix,
							 new_models = "pca_svm,AlexNet,umap_svm".split(","), new_rows = 2)
	for split_strategy in "personwise,tissuewise".split(","):
		main_hatchwise(os.path.join(res_Heatmap_root, split_strategy + "res_stat-data_all.csv"),
					   os.path.join(res_Heatmap_root, split_strategy + "res_stat-data_GBM.csv"),
					   os.path.join(split_strategy), skiprows = 2)


if __name__ == '__main__':
	main()
# print(plt.colormaps())
# main(res_stat_dir)

# main_hatchwise()
