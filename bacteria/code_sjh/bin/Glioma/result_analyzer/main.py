import os
import shutil

import roc_plot
import res_stat
import numpy
import val_res_heatmap

from bacteria.code_sjh.Core.basic_functions.path_func import getRootPath

projectroot = getRootPath("Raman_dl_ml")


def main_roc_stat(res_dir_base = "811_pointwise_upsampling", nets = None, dl = True, cal_conf_from_auc = True):
	res_root_up = "dl" if dl else "ml"

	res_root = os.path.join(projectroot, r"results\glioma\{}\{}".format(res_root_up, res_dir_base))
	# res_root = os.path.join(projectroot, r"results\glioma\ml\umap_bak\person_wise")
	res_Heatmap_root = os.path.join(projectroot, "results", "glioma", "Heatmap", "20240521", res_dir_base, "merge")
	if not os.path.isdir(res_Heatmap_root):
		os.makedirs(res_Heatmap_root)
	mode = "test" if dl else "val"
	statfiles = []
	for dirname in os.listdir(res_root):
		if os.path.isfile(os.path.join(res_root, "models.txt")):
			array = numpy.loadtxt(os.path.join(res_root, "models.txt"), dtype = "str", delimiter = ",", ndmin = 1)
			nets_ = list(array)
		elif nets is not None:
			nets_ = nets
		else:
			if dl:
				nets_ = ["AlexNet"]
			else:
				nets_ = "pca_svm,umap_svm".split(",")
		if not dirname.startswith("data"):
			continue

		roc_plot.main(os.path.join(res_root, dirname), nets = nets_, mode = mode,
					  sv_dir = os.path.join(res_Heatmap_root, "roc"))
		dst_file = res_stat.main(os.path.join(res_root, dirname), nets = nets_, mode = mode,
								 cal_conf_from_auc = cal_conf_from_auc)
		statfiles.append(dst_file)
		shutil.copy(dst_file, os.path.join(res_Heatmap_root, os.path.basename(dst_file)))
	print("static result saved in {}".format(res_Heatmap_root))


def main_Heatmap(dataroot = "811_pointwise_upsampling"):
	from matplotlib import colors
	res_Heatmap_root = os.path.join(projectroot, "results", "glioma", "Heatmap", "20240521", dataroot, "merge")
	l_prefix = "res_stat-data_all,res_stat-data_batch123,res_stat-data_GBM".split(",")
	for prefix in l_prefix:
		val_res_heatmap.merge_stat_files(res_Heatmap_root, res_Heatmap_root + "_" + prefix + ".csv", prefix,
										 new_models = "pca_svm,AlexNet,umap_svm".split(","), new_rows = 2)
	# for res_stat_dir in os.listdir(res_Heatmap_root):
	# 	res_stat_dir = os.path.join(res_Heatmap_root, res_stat_dir)
	# 	if not os.path.isdir(res_stat_dir) or not res_stat_dir.endswith("wise"): continue
	# 	for prefix in l_prefix:
	# 		val_res_heatmap.merge_stat_files(res_stat_dir, os.path.join(res_stat_dir + prefix + ".csv"), prefix,
	# 										 new_models = "pca_svm,AlexNet,umap_svm".split(","), new_rows = 2)
	# for split_strategy in "person_wise,tissue_wise,point_wise".split(","):
	# for split_strategy in ["point_wise"]:
	val_res_heatmap.main_hatchwise(os.path.join(res_Heatmap_root + "_res_stat-data_batch123.csv"),
								   os.path.join(res_Heatmap_root + "_res_stat-data_GBM.csv"),
								   os.path.join(res_Heatmap_root + "_plot_res"), skiprows = 2,
								   norm = colors.Normalize(0.5, 1, clip = True))


if __name__ == '__main__':
	data = "9fold_91_pointwise_upsampling_brnf"
	# data = "5fold_82_tissuewise_oversampling_brnf"
	# data = "tissuewise"
	main_roc_stat(data, dl = True, cal_conf_from_auc = True)
	main_roc_stat(data, dl = False, cal_conf_from_auc = True)
	main_Heatmap(data)
