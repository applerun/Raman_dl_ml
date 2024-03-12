import json
import shutil
import warnings
import csv
from sklearn.metrics import accuracy_score

# import pysnooper
from bacteria.code_sjh.utils.Validation.validation import *
from bacteria.code_sjh.utils.RamanData import data_leak_check_by_filename, Raman_depth_gen, get_dict_str
from bacteria.code_sjh.Core.basic_functions.mpl_func import *
from bacteria.code_sjh.ML import PCA, LDA, basic_SVM, UMAP
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
	train_data, train_label = [np.squeeze(x) for x in train_db.Ramans], [x for x in
	                                                                     train_db.labels]

	model.fit(train_data, train_label)

	# val_data, val_label = [np.squeeze(x.numpy()) for x in val_db.Ramans], [x.item() for x in
	#                                                                        val_db.labels]
	val_data, val_label = [np.squeeze(x) for x in val_db.Ramans], [x for x in
	                                                               val_db.labels]
	val_prob = model.predict_proba(val_data)
	val_acc = model.score(val_data, val_label)
	val_pred = model.predict(val_data)

	# test_data, test_label = [np.squeeze(x.numpy()) for x in test_db.Ramans], [x.item() for x in
	#                                                                           test_db.labels]
	test_data, test_label = [np.squeeze(x) for x in test_db.Ramans], [x for x in test_db.labels]
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


def train_modellist(
		dataroot,
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
	if recorddir is None:
		recorddir = "Record_" + time.strftime("%Y-%m-%d-%H_%M_%S")
		recorddir = os.path.join(projectroot, "results", "tissue_dl", recorddir)
	if modellist is None:
		modellist = [basic_SVM(), basic_SVM(PCA(n_components = 10)), basic_SVM(LDA(n_components = 1))]

	k_split = db_cfg["k_split"]

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
		f.write("\n")
		with open(os.path.join(recordsubdir, "db_cfg.txt"), "w") as sf:
			sf.write(get_dict_str(db_cfg) + "\n")
		writer.writerow(["n", "k", "val_acc", "val_AUC"])
		conf_m_v = None
		i = 0  # 实验进度计数
		valaccs, vaucs = [], []
		test_db_ = test_db
		for n in range(n_iter):
			for k in range(k_split):

				sfpath = sfname + str(n) + ".csv"
				train_db = raman(**db_cfg, mode = "train", k = k, sfpath = sfpath)
				val_db = raman(**db_cfg, mode = "val", k = k, sfpath = sfpath)
				if db_cfg["t_v_t"][2] > 0 or test_db_ is not None: warnings.warn(
					"val_db is not needed for machine learning")

				if db_cfg["t_v_t"][2] == 0 and test_db_ is None:
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
		assert len(valaccs) == len(vaucs) == n_iter * k_split, "valaccs:{}\n,vaucs:{}\n,n*k = {}*{}".format(valaccs,
		                                                                                                    vaucs,
		                                                                                                    n_iter,
		                                                                                                    k_split)

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
		# modellist = [basic_SVM(), basic_SVM(PCA(n_components = 10)), basic_SVM(LDA(n_components = 1))]
		modellist = [basic_SVM(PCA(n_components = 10)),
		             basic_SVM(UMAP(n_neighbors = 200,
		                            # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
		                            n_components = 3,
		                            n_jobs = 1,
		                            # default 2, The dimension of the space to embed into.
		                            metric = 'euclidean',
		                            # default 'euclidean', The metric to use to compute distances in high dimensional space.
		                            n_epochs = 1000,
		                            # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
		                            learning_rate = 1.0,
		                            # default 1.0, The initial learning rate for the embedding optimization.
		                            init = 'spectral',
		                            # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
		                            min_dist = 0.1,
		                            # default 0.1, The effective minimum distance between embedded points.
		                            spread = 1.0,
		                            # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
		                            low_memory = False,
		                            # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
		                            set_op_mix_ratio = 1.0,
		                            # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
		                            local_connectivity = 1,
		                            # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
		                            repulsion_strength = 1.0,
		                            # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
		                            negative_sample_rate = 5,
		                            # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
		                            transform_queue_size = 4.0,
		                            # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
		                            a = None,
		                            # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
		                            b = None,
		                            # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
		                            random_state = 42,
		                            # default: None, If int, random_state is the seed used by the random number generator;
		                            metric_kwds = None,
		                            # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
		                            angular_rp_forest = False,
		                            # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
		                            target_n_neighbors = -1,
		                            # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
		                            # target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different.
		                            # target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
		                            # target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
		                            transform_seed = 42,
		                            # default 42, Random seed used for the stochastic aspects of the transform operation.
		                            verbose = False,
		                            # default False, Controls verbosity of logging.
		                            unique = False,
		                            # default False, Controls if the rows of your data should be uniqued before being embedded.
		                            )), ]
		if not os.path.isdir(dataroot):
			continue
		db_cfg = dict(  # 数据集设置
			dataroot = dataroot,
			backEnd = ".csv",
			# backEnd = ".asc",
			t_v_t = [0.6, 0.4, 0.0],
			LoadCsvFile = readdatafunc,
			k_split = 9,
			transform = Process.process_series([  # 设置预处理流程
				preprocess,
				Process.sg_filter(),
				Process.norm_func(), ]
			))
		recorddir = os.path.join(recordroot, ele)
		path2labelfunc = path2func_generator(num2label)
		train_modellist(dataroot, db_cfg = db_cfg, raman = raman, modellist = modellist, recorddir = recorddir,
		                path2labelfunc = path2labelfunc,
		                sfname = "Raman_{}_".format("personwise" if personwise else "tissuewise"), n_iter = 1, )


def main_onesrc(datasplit = "personwise",
                dataroot_ = None):
	glioma_data_root = os.path.join(projectroot, "data", "脑胶质瘤")
	if dataroot_ is None:
		dataroot_ = os.path.join(glioma_data_root, "labeled_data\data_all_labeled")
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
	                 raman = Raman_depth_gen(3, 3) if datasplit == "pointwise" else Raman_dirwise,
	                 record_info = os.path.basename(dataroot_dst) + "_" + datasplit,
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
		if dir == "data_GBM_labeled": continue
	# for dir in ["data_GBM_labeled"]:

		dir_abs = os.path.join(glioma_data_root, "labeled_data", dir)
		if not os.path.isdir(dir_abs) or not dir.startswith("data") or dir.endswith(
				("personwise", "failed")) or "indep" in dir:
			continue
		# main_onesrc(datasplit = "personwise", dataroot_ = dir_abs)
		main_onesrc(datasplit = "pointwise", dataroot_ = dir_abs)
