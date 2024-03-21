import copy
import json
import shutil
import warnings
import csv

import sklearn.decomposition
import umap
from sklearn.metrics import accuracy_score

# import pysnooper
from bacteria.code_sjh.utils.Validation.validation import *
from bacteria.code_sjh.utils.RamanData import data_leak_check_by_filename, Raman_depth_gen, get_dict_str
from bacteria.code_sjh.Core.basic_functions.mpl_func import *
from bacteria.code_sjh.ML import gen_reduction_estimator, basic_SVM, PCA
from bacteria.code_sjh.bin.Glioma.Classify_dl.record_func import plt_res, npsv, heatmap
from bacteria.code_sjh.bin.Glioma.data_handler.label_handler import get_infos, path2func_generator
from bacteria.code_sjh.utils import Process
from bacteria.code_sjh.Core.basic_functions.path_func import getRootPath
from bacteria.code_sjh.Core.basic_functions.fileReader import getRamanFromFile

projectroot = getRootPath("Raman_dl_ml")
coderoot = getRootPath("code_sjh")
logfile = os.path.join(projectroot, "log", "glioma", "ML_classification", os.path.basename(__file__),
					   time.strftime("%Y-%m-%d-%H_%M_%S") + ".txt")
# @pysnooper.snoop(logfile, prefix = "--*--")
if not os.path.isdir(os.path.dirname(logfile)):
	os.makedirs(os.path.dirname(logfile))
with open(logfile, "w") as f:
	pass

readdatafunc = getRamanFromFile(
	# 定义读取数据的函数
	wavelengthstart = 39, wavelengthend = 1810, delimeter = None, dataname2idx = {"Wavelength": 2, "Intensity": 6}
)


def evaluate_all_by_score(Y_pred,
						  Y_probe,
						  Y_true):
	acc = accuracy_score(Y_true, Y_probe)
	num_classes = len(Y_probe[0])
	label2auc = {}
	label2roc = {}
	for i in range(num_classes):
		label_true = np.equal(Y_true, i).astype(int)
		score = Y_probe[:, i]
		# label2auc[i] = auc(frp, tpr)
		frp, tpr, thresholds = roc_curve(label_true, score)

		label2roc[i] = (frp, tpr, thresholds)
		label2auc[i] = auc(frp, tpr)
	conf_m_val = confusion_matrix(Y_probe, Y_pred)


def train_classification_model(
		model: basic_SVM,
		train_db,
		val_db,
		test_db,
		modelname = None,
):
	# train_data, train_label = [np.squeeze(x.numpy()) for x in train_db.Ramans], [x.item() for x in
	#                                                                              train_db.labels]
	if len(train_db) < len(val_db) * 2:
		warnings.warn("the ratio of train_db{} vs val_db{} is low, please recheck data split or function usage".format(
			len(train_db), len(val_db)))

	train_data, train_label = [np.squeeze(x) for x in train_db.Ramans], [x for x in train_db.labels]
	model.fit(train_data, train_label)
	res_val = evaluate_all_ml(model, val_db)

	test_val = evaluate_all_ml(model, test_db)
	res = dict(
		best_acc = res_val["acc"],
		res_test = test_val, res_val = res_val,  # 验证集所有指标
	)
	return res


def train_modellist(
		db_cfg,
		raman = RamanDatasetCore,
		# 设置读取数据集的DataSet
		# 设置k叠交叉验证的k值
		modellist = None,
		recorddir = None,
		path2labelfunc = None,
		test_db = None,
		sfname = "Raman_",
		n_iter = 1
		# 交叉验证重复次数
):
	warned = False
	if recorddir is None:
		recorddir = "Record_" + time.strftime("%Y-%m-%d-%H_%M_%S")
		recorddir = os.path.join(projectroot, "results", "tissue_dl", recorddir)

	k_split = db_cfg["k_split"]

	if not os.path.isdir(recorddir):
		os.makedirs(recorddir)
	with open(os.path.join(recorddir, "db_cfg.txt"), "w") as sf:
		sf.write(get_dict_str(db_cfg) + "\n")

	for n_m, model in enumerate(modellist):

		print("model:{}, {}/{}".format(model.model_name, n_m, len(modellist)))
		recordsubdir = os.path.join(recorddir,
									"Record" + model.model_name)  # + time.asctime().replace(":", "-").replace(" ", "_"))  # 每个模型一个文件夹保存结果
		if not os.path.isdir(recordsubdir):
			os.makedirs(recordsubdir)

		recordfile = recordsubdir + ".csv"  # 记录训练的配置和结果

		f = open(recordfile, "w", newline = "")
		writer = csv.writer(f)
		f.write("\n")
		writer.writerow(["n", "k", "val_acc", "test_acc", "val_AUC", "test_AUC"])
		valaccs, testaccs, bepochs, vaucs, taucs = [], [], [], [], []
		conf_m_v = None
		conf_m_t = None
		i = 0  # 实验进度计数

		# test_db_ = test_db
		for n in range(n_iter):
			for k in range(k_split):

				sfpath = sfname + str(n) + ".csv"
				train_db = raman(**db_cfg, mode = "train", k = k, sfpath = sfpath)
				val_db = raman(**db_cfg, mode = "val", k = k, sfpath = sfpath)
				test_db = raman(**db_cfg, mode = "test", k = k, sfpath = sfpath)
				if test_db is None:
					test_db = raman(**db_cfg, mode = "test")
				if len(test_db) == 0:
					test_db = val_db
				if db_cfg["t_v_t"][2] > 0 or test_db is not None:
					if warned == False:
						warnings.warn(
							"test_db is used for machine learning:{}".format(db_cfg["t_v_t"][2]))
						warned = True
				# if db_cfg["t_v_t"][2] == 0 and test_db is None:
				# 	test_db = val_db
				# elif test_db is None:
				# 	test_db = raman(**db_cfg, mode = "test", k = k, sfpath = sfpath)

				l = data_leak_check_by_filename((train_db, val_db))
				if len(l) > 0:
					warnings.warn("data leak warning:{} \n {}".format(len(l) / len(val_db), l))

				# __________________label_______________________
				# label_RamanData(train_db, path2labelfunc, name2label)
				# label_RamanData(val_db, path2labelfunc, name2label)
				if conf_m_v is None:
					conf_m_v = np.zeros((train_db.numclasses, train_db.numclasses))
					conf_m_t = np.zeros((train_db.numclasses, train_db.numclasses))
				assert len(val_db) > 0, str(val_db.sfpath) + ":" + str(val_db.RamanFiles)

				res = train_classification_model(model, train_db, val_db, test_db)

				pltdir = os.path.join(recordsubdir, "n-{}-k-{}".format(n, k))
				if not os.path.isdir(pltdir):
					os.makedirs(pltdir)
				v_acc, t, auc_val, auc_test = res["best_acc"], res["res_test"]["acc"], \
					np.mean(list(res["res_val"]["label2auc"].values())), \
					np.mean(list(res["res_test"]["label2auc"].values()))
				writer.writerow([n, k, v_acc, t, auc_val, auc_test])
				valaccs.append(v_acc)
				testaccs.append(t)
				vaucs.append(auc_val)
				taucs.append(auc_test)
				i += 1
				print(i, "/", n_iter * k_split)

				plt_res(pltdir, res, val_db, informations = None, mode = "ml")
				npsv(pltdir, res, val_db, mode = "ml")

				shape = res["res_val"]["confusion_matrix"].shape
				assert shape[0] == shape[1] == train_db.numclasses
				shape = res["res_test"]["confusion_matrix"].shape
				assert shape[0] == shape[1] == train_db.numclasses

				conf_m_v += res["res_val"]["confusion_matrix"]
				conf_m_t += res["res_test"]["confusion_matrix"]

		np.savetxt(os.path.join(recordsubdir, "val_confusion_matrix.csv"), conf_m_v, delimiter = ",")
		np.savetxt(os.path.join(recordsubdir, "test_confusion_matrix.csv"), conf_m_t, delimiter = ",")

		heatmap(conf_m_v, os.path.join(recordsubdir, "val_confusion_matrix.png"))
		heatmap(conf_m_v, os.path.join(recordsubdir, "val_confusion_matrix.png"))
		# train_db.shufflecsv()
		assert len(valaccs) == len(vaucs) == n_iter * k_split, "valaccs:{}\n,vaucs:{}\n,n*k = {}*{}".format(valaccs,
																											vaucs,
																											n_iter,
																											k_split)

		va = np.mean(np.array(valaccs)).__str__() + "+-" + np.std(np.array(valaccs)).__str__()
		ta = np.mean(np.array(testaccs)).__str__() + "+-" + np.std(np.array(testaccs)).__str__()
		auc_av = np.mean(np.array(vaucs)).__str__() + "+-" + np.std(np.array(vaucs)).__str__()
		auc_at = np.mean(np.array(taucs)).__str__() + "+-" + np.std(np.array(taucs)).__str__()

		writer.writerow(["mean", "std", va, ta, auc_av, auc_at])
		f.write("\n" + str(model))
		f.close()
		print("val acc", va)
		print("test acc", ta)


# 设置数据集分割
def main_one_datasrc(
		dataroot_ = os.path.join(projectroot, "data", "脑胶质瘤", "data_all"),
		info_file = os.path.join(projectroot, "data", "脑胶质瘤", "data_used\病例编号&分类结果2.xlsx"),
		raman = Raman_dirwise,
		record_info = None,
		personwise = True,
		db_cfg = None,
		record_root_basename = None,
):
	if db_cfg is None:
		db_cfg = {}
	num2ele2label = get_infos(info_file)
	eles = list(num2ele2label.values().__iter__().__next__().keys())
	if record_info is None:
		record_info = ""
	if len(record_info) > 0:
		record_info = record_info + "_"

	recordroot = os.path.join(projectroot, "results", "glioma", "ml")
	if record_root_basename is not None:
		recordroot = os.path.join(recordroot, record_root_basename)
	recordroot = os.path.join(recordroot, record_info + time.strftime("%Y-%m-%d-%H_%M_%S"))
	#   # TODO:根据数据存储方式选择合适的读取策略（Raman/Raman_dirwise)

	paras = [(e, pre) for e in eles for pre in [
		# Process.baseline_als(),
		Process.bg_removal_niter_fit(),
		# Process.bg_removal_niter_piecewisefit(),
	]]

	for ele, preprocess in paras:
		num2label = {}
		for k in num2ele2label.keys():
			num2label[k] = num2ele2label[k][ele]
		name2label = {"neg": 0, "pos": 1}
		dataroot = os.path.join(dataroot_, ele)
		# modellist = [basic_SVM(), basic_SVM(PCA(n_components = 10)), basic_SVM(LDA(n_components = 1))]
		modellist = [basic_SVM(gen_reduction_estimator(PCA,"pca", n_components = 10)),
					 basic_SVM(gen_reduction_estimator(umap.UMAP,"umap",n_neighbors = 200,
									# default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
									n_components = 3,
									# default 2, The dimension of the space to embed into.
									# n_jobs = 1,
									n_epochs = 1000,
									# default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
									# random_state = 42,
									# default: None, If int, random_state is the seed used by the random number generator;
									)), ]
		if not os.path.isdir(dataroot):
			continue

		db_cfg["dataroot"] = dataroot
		recorddir = os.path.join(recordroot, ele)
		path2labelfunc = path2func_generator(num2label)
		train_modellist(db_cfg = db_cfg, raman = raman, modellist = modellist, recorddir = recorddir,
						path2labelfunc = path2labelfunc,
						sfname = "Raman_{}_".format("personwise" if personwise else "tissuewise"), n_iter = 1, )


def main_onesrc(datasplit = "personwise",
				dataroot_ = None, db_cfg = None, record_root_basename = None):
	glioma_data_root = os.path.join(projectroot, "data", "脑胶质瘤")
	if dataroot_ is None:
		dataroot_ = os.path.join(glioma_data_root, r"labeled_data\data_all_labeled")
	# dataroot_ = os.path.join(glioma_data_root, "labeled_data\data_batch123_labeled")
	personwise = datasplit == "personwise"
	if personwise:
		from bacteria.code_sjh.bin.Glioma.data_handler.samplewise2personwise import rename_files_between
		dataroot_dst = dataroot_ + "_renamed_for_personwise"
		# rename_files_between(dataroot_dst, 3)
		if not os.path.isdir(dataroot_dst):
			if os.path.isdir(dataroot_dst + "_failed"):
				shutil.move(dataroot_dst + "_failed", dataroot_dst)
			else:
				shutil.copytree(dataroot_, dataroot_dst)
			try:
				rename_files_between(dataroot_dst, 3)
			except:
				warnings.warn("rename_failed")
				shutil.move(dataroot_dst, dataroot_dst + "_failed")
				dataroot_dst = dataroot_
	else:
		dataroot_dst = dataroot_

	info_file = os.path.join(projectroot, "data", "脑胶质瘤", "data_used\病例编号&分类结果2.xlsx")
	main_one_datasrc(dataroot_dst, info_file,
					 raman = Raman_depth_gen(2, 2) if datasplit == "pointwise" else Raman_dirwise,
					 record_info = os.path.basename(dataroot_dst) + "_" + datasplit, db_cfg = db_cfg,
					 record_root_basename = record_root_basename
					 )


# rename_files_between_undo(dataroot_dst, 3)


# try:
# 	main_one_datasrc(dataroot_dst, info_file)
# except:
# 	print("failed")
# finally:
# 	rename_files_between_undo(dataroot_dst, 3)


if __name__ == '__main__':
	glioma_data_root = os.path.join(projectroot, "data", "脑胶质瘤")

	preprocess_liu = Process.process_series([  # 文章预处理流程
		# Process.interpolator(400, 1800, 256),
		Process.bg_removal_niter_fit(),
		Process.sg_filter(window_length = 11),
		Process.norm_func(), ]
	)
	preprocess_bals = Process.process_series([  # 优化的预处理流程
		Process.interpolator(400, 1800, 512),
		Process.baseline_als(),
		Process.sg_filter(window_length = 21),
		Process.norm_func(), ]
	)

	db_cfg = dict(  # 数据集设置
		backEnd = ".csv",
		# backEnd = ".asc",
		t_v_t = [0.6, 0.2, 0.2],
		LoadCsvFile = readdatafunc,
		k_split = 8,
		transform = preprocess_liu,
		class_resampling = "over",
	)

	datasplit = "pointwise"

	record_root_basename = f"{db_cfg['k_split']}fold_{int(sum(db_cfg['t_v_t'][:2]) * 10)}{int(db_cfg['t_v_t'][2] * 10)}_{datasplit}_{'nore' if db_cfg['class_resampling'] is None else db_cfg['class_resampling']}sampling_brnf"

	for dir in os.listdir(os.path.join(glioma_data_root, "labeled_data")):
		if dir == "data_GBM_labeled": continue
		# for dir in ["data_GBM_labeled"]:

		dir_abs = os.path.join(glioma_data_root, "labeled_data", dir)
		if not os.path.isdir(dir_abs) or not dir.startswith("data") or "indep" in dir or \
				dir.endswith(("personwise", "failed")):
			continue
		# main_onesrc(datasplit = "personwise", dataroot_ = dir_abs)
		main_onesrc(datasplit = datasplit, dataroot_ = dir_abs, db_cfg = db_cfg,
					record_root_basename = record_root_basename)
