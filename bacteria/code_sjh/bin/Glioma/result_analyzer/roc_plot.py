import warnings

import numpy
import os
from scipy import interpolate
from numpy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score
import csv
from bacteria.code_sjh.Core.basic_functions.mpl_func import spectrum_vis_mpl

color1 = numpy.array([254, 194, 67, 150]) / 255
color4 = numpy.array([166, 206, 227, 150]) / 255

color_all = [color1, color4]


def cal_roc_macro(fpr: list, tpr: list, type = "macro"):
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


def rocRead_bi(src, mode = "test", pos_label = "pos"):
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
			f_all, t_all = [],[]
			for rocfile in rocfiles:
				f, t, thr = numpy.loadtxt(os.path.join(rocfile_dir, rocfile), delimiter = ",", skiprows = 1).T
				f_all.append(f)
				t_all.append(t)
			f, _, t = cal_roc_macro(f_all, t_all)
		fprs.append(f)
		tprs.append(t)
	fpr_macro, _,tpr_macro = cal_roc_macro(fprs, tprs)
	auc_macro = auc(fpr_macro,tpr_macro)
	numpy.savetxt(os.path.join(src, "roc_record.csv"), numpy.vstack((fpr_macro, tpr_macro)).T, delimiter = ",",
				  header = "fpr,tpr,auc={}".format(auc_macro),
				  comments = "")
	return fprs, tprs


def main_bi_class(dir, dst, mode = "test",
				  nets = None,
				  molecules = None):
	if nets is None:
		nets = ["Alexnet_Sun", "Resnet18", "Resnet34"]
	if not os.path.isabs(dst):
		dst = os.path.join(dir, dst)
	# if not os.path.isdir(dst):
	# 	os.makedirs(dst)

	if molecules is None:
		molecules = os.listdir(dir)

	for molecule in molecules:
		molecule_abs = os.path.join(dir, molecule)
		if not os.path.isdir(molecule_abs):
			continue

		for i, net in enumerate(nets):
			src = os.path.join(molecule_abs, "Record" + net)
			rocRead_bi(src)

	return


def main(dirname, nets = None):
	if nets is None:
		nets = ["Alexnet"]
	# ms = "IDH(M-1)@1p-19q(缺-1)@M(甲基化-1)@T(突变-1)@E(扩增-1)@7(+ 1)@10(- 1)@A(缺-1)@B(缺-1)"
	ms = "IDH(M-1)@1p-19q(缺-1)@M(甲基化-1)@T(突变-1)@E(扩增-1)@7+-10-@AB(共缺-1)"
	resultdir = os.path.join(r"D:\myPrograms\pythonProject\Raman_dl_ml\results\glioma\dl", dirname)
	dst_file = os.path.join(resultdir, "res_stat-" + dirname + ".csv")
	main_bi_class(resultdir, dst_file, nets = nets, molecules = ms.split("@"))


if __name__ == '__main__':
	main("2024-03-04-23_19_32")
