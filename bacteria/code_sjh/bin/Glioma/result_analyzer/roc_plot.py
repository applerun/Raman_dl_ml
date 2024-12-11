import copy
import shutil
import warnings
import matplotlib.pyplot as plt
import numpy
import os
from scipy import interpolate
from numpy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score
import csv
from bacteria.code_sjh.Core.basic_functions.mpl_func import spectrum_vis_mpl

import matplotlib as mpl

mpl.rcParams["font.family"] = 'Arial'  # 设置字体

mpl.rcParams["axes.unicode_minus"] = False  # 正常显示负号
color1 = numpy.array([254, 194, 67, 150]) / 255
color4 = numpy.array([166, 206, 227, 150]) / 255

color_all = [color1, color4]


def cal_roc_macro(fpr: list,
				  tpr: list,
				  type = "macro"):
	"""

	@param fpr: (l,x)
	@param tpr: (l,x)
	@param type: "macro" or "micro"
	@return: fpr:ndarray(x),tpr:ndarray(x),auc:float or None(unsupported type)
	"""
	if len(fpr) == 1:
		return fpr, tpr
	if type == "macro":
		all_fpr = numpy.unique(numpy.concatenate(fpr))
		all_tpr = None
		for i, fpr_i in enumerate(fpr):
			tpr_i = interp(all_fpr, fpr_i, tpr[i])
			all_tpr = tpr_i if all_tpr is None else numpy.vstack((all_tpr, tpr_i))
		mean_tpr = all_tpr.mean(axis = 0)
		return all_fpr, all_tpr, mean_tpr

	elif type == "micro":
		return None
	else:
		print("unsupported type:{}".format(type))
		return None


# def plot_roc(fpr, tpr, thresholds, axes):


def rocRead_bi(src,
			   mode = "test",
			   pos_label = "pos"):
	f"""
	read： src/*/roc/{mode}_{pos_label}_roc.csv (二分类) or src/*/roc/{mode}_?_roc.csv (多分类) 
	每个*对应的文件夹得到一个roc曲线（多分类的时候为平均roc）
	返回所有的曲线，并计算平均roc
	平均roc曲线保存在src/roc_record.csv
	@param src: 
	@param mode: 
	@param pos_label: 
	@return: 
	"""
	fprs = []
	tprs = []
	for dirs in os.listdir(src):
		dir_abs = os.path.join(src, dirs)
		if not os.path.isdir(dir_abs):
			continue
		rocfile_dir = os.path.join(dir_abs, "roc")
		if not os.path.isdir(rocfile_dir):
			warnings.warn("roc dir not found in {}".format(dir_abs))
			continue
		rocfiles = []
		for f in os.listdir(rocfile_dir):
			if f.endswith(".csv") and f.startswith(mode):
				rocfiles.append(f)
		# 2分类时取pos label的 ROC ，多分类取平均roc
		if len(rocfiles) == 0:
			warnings.warn("roc file not found in {}".format(rocfile_dir))
			continue
		elif len(rocfiles) == 1:
			warnings.warn("only one roc file in {}".format(rocfile_dir))
			continue
		elif len(rocfiles) == 2:
			rocfile_ = "{}_{}_roc.csv".format(mode, pos_label)
			if rocfile_ in rocfiles:
				rocfile = rocfile_
			else:
				warnings.warn("name of the file is not in a supported format:{} in {}".format(rocfiles, rocfile_dir))
				rocfile = None
				for file in rocfiles:
					if pos_label in file:
						rocfile = file
						break
				if rocfile is None:
					warnings.warn(
						"pos_label {} not found in {}".format(pos_label, rocfile_dir))
					return -1

			f, t, thr = numpy.loadtxt(os.path.join(rocfile_dir, rocfile), delimiter = ",", skiprows = 1).T
		else:
			f_all, t_all = [], []
			for rocfile in rocfiles:
				f, t, thr = numpy.loadtxt(os.path.join(rocfile_dir, rocfile), delimiter = ",", skiprows = 1).T
				f_all.append(f)
				t_all.append(t)
			f, _, t = cal_roc_macro(f_all, t_all)
		fprs.append(f)
		tprs.append(t)
	fpr_macro, tpr_all, tpr_macro = cal_roc_macro(fprs, tprs)
	auc_macro = auc(fpr_macro, tpr_macro)
	numpy.savetxt(os.path.join(src, "roc_record.csv"), numpy.vstack((fpr_macro, tpr_macro, tpr_all)).T, delimiter = ",",
				  header = "fpr,tpr,auc={}".format(auc_macro),
				  comments = "")
	return fpr_macro, tpr_all, tpr_macro


def plot_roc(fpr,
			 tpr,
			 title,
			 axes: plt.Axes = None,
			 show_auc = True,
			 color = color1,
			 name = None):
	if axes is None:
		fig, axes = plt.subplots(1)
	if show_auc:
		if name is None:
			name = ""
		if len(name) > 0:
			name = name + " "
		auc_res = auc(fpr, numpy.mean(tpr, axis = 0) if len(list(tpr.shape)) == 2 else tpr)
		name = name + "(AUC = {:.3f})".format(round(auc_res, 4))

	if name is None:
		if len(list(tpr.shape)) == 2:
			color_shadow = copy.deepcopy(color)
			color_shadow[3] = color_shadow[3] * 0.4
			spectrum_vis_mpl(tpr, fpr, ax = axes, line_color = color, shadow_color = color_shadow)
		else:
			axes.plot(fpr, tpr, color = color)
	else:
		if len(list(tpr.shape)) == 2:
			color_shadow = copy.deepcopy(color)
			color_shadow[3] = color_shadow[3] * 0.4
			spectrum_vis_mpl(tpr, fpr, ax = axes, name = name, line_color = color, shadow_color = color_shadow)
		else:
			axes.plot(fpr, tpr, color = color, label = name)
		axes.legend(loc = "lower right", fontsize = 10, frameon = False)
	for s in axes.spines:
		axes.spines[s].set_color(numpy.array([100, 100, 100]) / 255)
	axes.set_xlim(-0.005, 1)
	axes.set_ylim(0, 1.01)
	xfontdict = dict(fontsize = 15, fontstyle = "normal", fontweight = 400)
	yfontdict = dict(fontsize = 15, fontstyle = "normal", fontweight = 400)
	axes.set_xlabel("1 - Specificity", fontdict = xfontdict)
	axes.set_ylabel("Sensitivity", fontdict = yfontdict)
	# tickfontdict = dict(fontsize = 5, fontstyle = "normal", fontweight = 400)
	axes.tick_params(labelsize = 10)
	# axes.set_yticks(fontdict = tickfontdict)
	title_fontdict = dict(fontsize = 18, fontstyle = "normal", fontweight = 450)
	axes.set_title(title, fontdict = title_fontdict)
	return axes


def main_bi_class(dir,
				  mode = "test",
				  nets = None,
				  molecules = None,
				  positions = None,
				  net2axes = None,
				  sv_dir = None,
				  titles = None,
				  shadow = True
				  ):
	"""

	@param dir: src dir
	@param mode:
	@param nets:
	@param molecules:
	@param positions:
	@param net2axes:
	@param sv_dir: 将图片另存为某文件夹
	@return:
	"""
	if nets is None:
		nets = ["Alexnet_Sun", "Resnet18", "Resnet34"]

	# if not os.path.isdir(dst):
	# 	os.makedirs(dst)

	if molecules is None:
		molecules = os.listdir(dir)
	else:
		molecules = copy.deepcopy(molecules)
	if len(molecules) == len(positions):
		molecules2positions = dict(zip(molecules, positions))
	else:
		molecules2positions = None
	if titles is not None and len(titles) == len(positions):
		positions2titles = dict(zip(positions, titles))
	else:
		positions2titles = None
	net2molecule2roc = {net: {} for net in nets}
	unfoundmolecule = []
	for molecule in molecules:
		molecule_abs = os.path.join(dir, molecule)
		if not os.path.isdir(molecule_abs) or molecule.startswith(("_", ".")):
			unfoundmolecule.append(molecule)
			continue
		for i, net in enumerate(nets):
			src = os.path.join(molecule_abs, "Record" + net)
			try:

				fpr, tpr_a, tpr_m = rocRead_bi(src, mode = mode)
				net2molecule2roc[net][molecule] = (fpr, tpr_a) if shadow else (fpr, tpr_m)
			except:
				net2molecule2roc[net][molecule] = None
	for molecule in unfoundmolecule:
		molecules.remove(molecule)
	for net, molecule2roc in net2molecule2roc.items():
		num_row = int(numpy.ceil(numpy.sqrt(len(molecules))))
		axes: numpy.ndarray
		if net2axes is None:
			fig, axes = plt.subplots(num_row, num_row, dpi = 600, figsize = (9.6, 10),
									 )
			# , constrained_layout = True
			plt.subplots_adjust(wspace = 0.4, hspace = 0.6)
		else:
			axes = net2axes[net]
			fig = axes[0].figure

		axes = axes.flatten()
		allidx = list(range(len(axes)))
		for i_, molecule in enumerate(molecules):
			if molecules2positions is not None:
				i = molecules2positions[molecule]
			elif positions is not None:
				i = positions[i_]
			else:
				i = i_
			ax = axes[i]
			ax.text(-0.25, 1.15, chr(i_ + 97), transform = ax.transAxes, fontsize = 15, fontweight = "bold")
			try:
				title = molecule if positions2titles is None else positions2titles[i]
				plot_roc(*molecule2roc[molecule], title, axes = ax, show_auc = True,
						 name = "ROC",
						 color = color4 if "GBM" in dir else color1)
				allidx.remove(i)
			except:
				continue
		if net2axes is None:
			for i in allidx:
				fig.delaxes(axes[i])
		fig.savefig(os.path.join(dir, net + "_roc.tif"))
		if sv_dir is not None:
			short_dir_base = os.path.basename(dir).split("_")
			short_dir_base = "_".join(short_dir_base[:min(len(short_dir_base), 3)])
			if not os.path.isdir(os.path.join(sv_dir, short_dir_base)):
				os.makedirs(os.path.join(sv_dir, short_dir_base))
			shutil.copy(os.path.join(dir, net + "_roc.tif"), os.path.join(sv_dir, short_dir_base, net + "_roc.tif"))
		plt.close(fig)
	return net2molecule2roc


def main(dirname,
		 nets = None,
		 mode = "test", sv_dir = None):
	if nets is None:
		nets = ["Alexnet"]
	if sv_dir is not None:
		if not os.path.isdir(sv_dir):
			os.makedirs(sv_dir)
	# ms = "IDH(M-1)@1p-19q(缺-1)@M(甲基化-1)@T(突变-1)@E(扩增-1)@7(+ 1)@10(- 1)@A(缺-1)@B(缺-1)"
	ms = "IDH(M-1)@1p19q(缺-1)@M(甲基化-1)@T(突变-1)@E(扩增-1)@7+-10-@AB(共缺-1)"
	resultdir = dirname
	titles = "IDH@1p/19q@MGMT@TERT@EGFR@Chromosome 7/10@CDKN2A/B".split("@")
	main_bi_class(resultdir, nets = nets, molecules = ms.split("@"), positions = [0, 1, 3, 4, 5, 6, 7], mode = mode,
				  sv_dir = sv_dir, titles = titles)


if __name__ == '__main__':
	from bacteria.code_sjh.Core.basic_functions.path_func import getRootPath

	projectroot = getRootPath("Raman_dl_ml")
	# root = os.path.join(projectroot, "data", "脑胶质瘤")
	res_root = os.path.join(projectroot, r"results\glioma\ml")
	# res_root = os.path.join(projectroot, r"results\glioma\dl")
	for dirname in os.listdir(res_root):

		if not dirname.startswith("data"):
			continue
		main(os.path.join(res_root, dirname), nets = ["svm", "pca_svm", "lda_svm"], mode = "val")
# main(os.path.join(res_root, dirname), nets = ["AlexNet"], mode = "test")
