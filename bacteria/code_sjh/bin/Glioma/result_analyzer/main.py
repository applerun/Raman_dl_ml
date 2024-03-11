import os
import shutil

import roc_plot
import res_stat
import numpy
import val_res_heatmap

from bacteria.code_sjh.Core.basic_functions.path_func import getRootPath

projectroot = getRootPath("Raman_dl_ml")


def main_roc_stat():
	res_root = os.path.join(projectroot, r"results\glioma\ml\svm_pca_umap_811_pointwise")
	# res_root = os.path.join(projectroot, r"results\glioma\ml\umap_bak\person_wise")
	res_Heatmap_root = os.path.join(projectroot, "results", "glioma", "Heatmap", "20240318", "person_wise")
	if not os.path.isdir(res_Heatmap_root):
		os.makedirs(res_Heatmap_root)
	statfiles = []
	for dirname in os.listdir(res_root):
		if os.path.isfile(os.path.join(res_root, "models.txt")):
			array = numpy.loadtxt(os.path.join(res_root, "models.txt"), dtype = "str", delimiter = ",", ndmin = 1)
			nets = list(array)
		if not dirname.startswith("data"):
			continue
		roc_plot.main(os.path.join(res_root, dirname), nets = nets, mode = "val")
		dst_file = res_stat.main(os.path.join(res_root, dirname), nets = nets, mode = "val")
		statfiles.append(dst_file)
		shutil.copy(dst_file, os.path.join(res_Heatmap_root, os.path.basename(dst_file)))


def main_Heatmap():
	from matplotlib import colors
	res_Heatmap_root = os.path.join(projectroot, "results", "glioma", "Heatmap", "20240318","9fold91")
	l_prefix = "res_stat-data_all,res_stat-data_GBM,res_stat-data_batch123".split(",")
	for res_stat_dir in os.listdir(res_Heatmap_root):
		res_stat_dir = os.path.join(res_Heatmap_root, res_stat_dir)
		if not os.path.isdir(res_stat_dir) or not res_stat_dir.endswith("wise"): continue
		for prefix in l_prefix:
			val_res_heatmap.merge_stat_files(res_stat_dir, os.path.join(res_stat_dir + prefix + ".csv"), prefix,
			                                 new_models = "pca_svm,AlexNet,umap_svm".split(","), new_rows = 2)
	# for split_strategy in "person_wise,tissue_wise,point_wise".split(","):
	for split_strategy in ["811point_wise"]:
		val_res_heatmap.main_hatchwise(os.path.join(res_Heatmap_root, split_strategy + "res_stat-data_all.csv"),
		                               os.path.join(res_Heatmap_root, split_strategy + "res_stat-data_GBM.csv"),
		                               os.path.join(res_Heatmap_root + "_plot_res", split_strategy), skiprows = 2,
		                               norm = colors.Normalize(0.5, 1, clip = True))


if __name__ == '__main__':
	# main_roc_stat()
	main_Heatmap()
