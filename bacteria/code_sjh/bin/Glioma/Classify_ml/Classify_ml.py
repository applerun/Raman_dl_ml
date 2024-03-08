import os
import shutil
import warnings
import csv
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score,accuracy_score,auc

import pysnooper
from bacteria.code_sjh.utils.Validation.validation import *
from bacteria.code_sjh.utils.RamanData import getRamanFromFile, Raman_depth_gen, data_leak_check_by_filename
from bacteria.code_sjh.Core.basic_functions.mpl_func import *
from bacteria.code_sjh.ML.Demension.traditional import PCA, LDA, basic_SVM
from bacteria.code_sjh.bin.Glioma.Classify_dl.record_func import plt_res, npsv, heatmap
from bacteria.code_sjh.bin.Glioma.data_handler.label_handler import get_infos, path2func_generator
from bacteria.code_sjh.utils import Process
from bacteria.code_sjh.Core.basic_functions.path_func import getRootPath


projectroot = getRootPath("Raman_dl_ml")
coderoot = getRootPath("code_sjh")
logfile = os.path.join(projectroot, "log", "glioma", "ML_classification", os.path.basename(__file__),
                       time.strftime("%Y-%m-%d-%H_%M_%S") + ".txt")
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
	train_data, train_label = [np.squeeze(x.numpy()) for x in train_db.Ramans], [x.item() for x in
	                                                                             train_db.labels]
	model.fit(train_data, train_label)

	val_data, val_label = [np.squeeze(x.numpy()) for x in val_db.Ramans], [x.item() for x in
	                                                                       val_db.labels]
	val_prob = model.predict_proba(val_data)
	val_acc = model.score(val_data, val_label)
	val_pred = model.predict(val_data)

	test_data, test_label = [np.squeeze(x.numpy()) for x in test_db.Ramans], [x.item() for x in
	                                                                          test_db.labels]
	test_prob = model.predict_proba(test_data)
	test_acc = model.score(test_data, test_label)
	test_pred = model.predict(test_data)

	label2auc = {}
	label2roc = {}
	conf_m_val = confusion_matrix(val_label, model.predict(val_data))
	for i in range(train_db.num_classes()):
		label_true = np.equal(val_label, i).astype(int)
		score = val_prob[:, i]
		# label2auc[i] = auc(frp, tpr)
		frp, tpr, thresholds = roc_curve(label_true, score)
		label2roc[i] = (frp, tpr, thresholds)
		label2auc[i] = auc(frp, tpr)
	res_val = dict(
		acc = val_acc, label2roc = label2roc, label2auc = label2auc, confusion_matrix = conf_m_val
	)
	res = dict(
		val_acc = val_acc,
		res_val = res_val,  # 验证集所有指标
		# acc：float正确率, loss:float,
		# label2roc:dict 各个label的ROC, label2auc:dict 各个label的AUC, confusion_matrix:np.ndarray 混淆矩阵
	)
	return res


@pysnooper.snoop(logfile, prefix = "--*--")
def train_modellist(
		dataroot,
		db_cfg = None,
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
	if recorddir is None:
		recorddir = "Record_" + time.strftime("%Y-%m-%d-%H_%M_%S")
		recorddir = os.path.join(projectroot, "results", "tissue_dl", recorddir)
	if modellist is None:
		modellist = [basic_SVM(), basic_SVM(PCA(n_components = 10)), basic_SVM(LDA(n_components = 1))]
	if db_cfg is None:
		db_cfg = dict(  # 数据集设置
			dataroot = dataroot,
			backEnd = ".csv",
			# backEnd = ".asc",
			t_v_t = [0.8, 0.2, 0.0],
			LoadCsvFile = readdatafunc,
			k_split = 6,
			transform = Process.process_series([  # 设置预处理流程
				Process.interpolator(),
				# Process.baseline_als(),
				Process.bg_removal_niter_fit(),
				# Process.bg_removal_niter_piecewisefit(),
				Process.sg_filter(),
				Process.norm_func(), ]
			)
		)
	if db_cfg["t_v_t"][2] > 0: warnings.warn("val_db is not needed for machine learning")

	k_split = db_cfg["k_split"]
	valaccs, vaucs = [], []

	if not os.path.isdir(recorddir):
		os.makedirs(recorddir)
	for model in modellist:
		recordsubdir = os.path.join(recorddir,
		                            "Record" + model.__name__)  # + time.asctime().replace(":", "-").replace(" ", "_"))  # 每个模型一个文件夹保存结果
		if not os.path.isdir(recordsubdir):
			os.makedirs(recordsubdir)
		recordfile = recordsubdir + ".csv"  # 记录训练的配置和结果

		f = open(recordfile, "w", newline = "")
		writer = csv.writer(f)
		f.write(db_cfg.__str__() + "\n")
		writer.writerow(["n", "k", "val_acc", "val_AUC"])
		conf_m_v = None
		i = 0  # 实验进度计数
		for n in range(n_iter):
			for k in range(k_split):

				sfpath = sfname + str(n) + ".csv"
				train_db = raman(**db_cfg, mode = "train", k = k, sfpath = sfpath)
				val_db = raman(**db_cfg, mode = "val", k = k, sfpath = sfpath)
				if db_cfg["t_v_t"][2] == 0 and test_db is None:
					test_db = val_db
				elif test_db is None:
					test_db = raman(**db_cfg, mode = "test", k = k, sfpath = sfpath)
				l = data_leak_check_by_filename((train_db, val_db))
				if len(l) > 0:
					warnings.warn("data leak warning:{} \n {}".format(len(l) / len(val_db), l))
				res = train_classification_model(model, train_db, val_db, test_db)

				pltdir = os.path.join(recordsubdir, "n-{}-k-{}".format(n, k))
				val_acc = res["val_acc"]
				valaccs.append(val_acc)
				if conf_m_v is None:
					conf_m_v = np.zeros((train_db.numclasses, train_db.numclasses))
				conf_m_v += res["res_val"]["confusion_matrix"]

				auc_m = np.mean(list(res["res_val"]["label2auc"].values()))
				vaucs.append(auc_m)
				writer.writerow([n, k, val_acc, auc_m])
				i += 1
				print(i, "/", 1 * k_split)

				if not os.path.isdir(pltdir):
					os.makedirs(pltdir)
				plt_res(pltdir, res, val_db, informations = None, mode = "ml")
				npsv(pltdir, res, val_db, mode = "ml")

				shape = res["res_val"]["confusion_matrix"].shape
				assert shape[0] == shape[1] == train_db.numclasses
				conf_m_v += res["res_val"]["confusion_matrix"]

		np.savetxt(os.path.join(recordsubdir, "val_confusion_matrix.csv"), conf_m_v, delimiter = ",")
		heatmap(conf_m_v, os.path.join(recordsubdir, "val_confusion_matrix.png"))
		# train_db.shufflecsv()

		ta = np.mean(np.array(valaccs)).__str__() + "+-" + np.std(np.array(valaccs)).__str__()
		auc_av = np.mean(np.array(vaucs)).__str__() + "+-" + np.std(np.array(vaucs)).__str__()

		writer.writerow(["mean", "std", ta, auc_av])
		f.write("\n" + str(model))
		f.close()

		print("val acc", ta)


# 设置数据集分割
def main_one_datasrc(
		dataroot_ = os.path.join(projectroot, "data", "脑胶质瘤", "data_all"),
		info_file = os.path.join(projectroot, "data", "脑胶质瘤", "data_used\病例编号&分类结果2.xlsx"),
		raman = Raman_dirwise,
		record_info = None,
		personwise = True,
):
	num2ele2label = get_infos(info_file)
	eles = list(num2ele2label.values().__iter__().__next__().keys())
	if record_info is None:
		record_info = ""
	if len(record_info) > 0:
		record_info = record_info + "_"
	recordroot = os.path.join(projectroot, "results", "glioma", "ml")
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
		modellist = [basic_SVM(), basic_SVM(PCA(n_components = 10)), basic_SVM(LDA(n_components = 1))]
		if not os.path.isdir(dataroot):
			continue
		db_cfg = dict(  # 数据集设置
			dataroot = dataroot,
			backEnd = ".csv",
			# backEnd = ".asc",
			t_v_t = [0.8, 0.2, 0.0],
			LoadCsvFile = readdatafunc,
			k_split = 5,
			transform = Process.process_series([  # 设置预处理流程
				Process.sg_filter(),
				preprocess,
				Process.norm_func(), ]
			))
		recorddir = os.path.join(recordroot, ele)
		path2labelfunc = path2func_generator(num2label)
		train_modellist(dataroot, db_cfg = db_cfg, raman = raman, modellist = modellist, recorddir = recorddir,
		                path2labelfunc = path2labelfunc,
		                sfname = "Raman_{}_".format("personwise" if personwise else "tissuewise"), n_iter = 1, )


def main_onesrc(personwise = True,
                dataroot_ = None):
	glioma_data_root = os.path.join(projectroot, "data", "脑胶质瘤")
	if dataroot_ is None:
		dataroot_ = os.path.join(glioma_data_root, "labeled_data\data_all_labeled")
	# dataroot_ = os.path.join(glioma_data_root, "labeled_data\data_batch123_labeled")

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
	main_one_datasrc(dataroot_dst, info_file, raman = Raman_dirwise,
	                 record_info = os.path.basename(dataroot_dst) + ("person_wise" if personwise else "tissue_wise"),
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
	for dir in os.listdir(os.path.join(glioma_data_root, "labeled_data")):
		# for dir in ["data_GBM_labeled"]:
		dir_abs = os.path.join(glioma_data_root, "labeled_data", dir)
		if not os.path.isdir(dir_abs) or not dir.startswith("data") or dir.endswith(("personwise", "failed")):
			continue

		main_onesrc(personwise = False, dataroot_ = dir_abs)
		main_onesrc(personwise = True, dataroot_ = dir_abs)
