from sklearn import svm
import os, time, csv
import numpy as np
from bacteria.code_sjh.utils.RamanData import Raman, projectroot, getRamanFromFile, Raman_dirwise
from bacteria.code_sjh.utils import Process
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
import seaborn
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA


def heatmap(matrix,
            path):
	cm_fig, cm_ax = plt.subplots()
	seaborn.heatmap(matrix, annot = True, cmap = "Blues", ax = cm_ax)
	cm_ax.set_title('confusion matrix')
	cm_ax.set_xlabel('predict')
	cm_ax.set_ylabel('true')
	cm_fig.savefig(path)
	plt.close(cm_fig)


def preprocess_PCA(x_train,
                   y_train,
                   x_test,
                   y_test):
	pca = PCA(n_components = 2)
	pca.fit(x_train)
	x_train = pca.transform(x_train)
	x_test = pca.transform(x_test)
	return x_train, y_train, x_test, y_test


def preprocess_LDA(x_train,
                   y_train,
                   x_test,
                   y_test):
	lda = LDA(n_components = 1)
	lda.fit(x_train, y_train)
	x_train = lda.transform(x_train)
	x_test = lda.transform(x_test)
	return x_train, y_train, x_test, y_test


if __name__ == '__main__':
	raman = Raman_dirwise
	k_split = 6
	readdatafunc0 = getRamanFromFile(wavelengthstart = 390, wavelengthend = 1810,
	                                 dataname2idx = {"Wavelength": 0, "Column": 2, "Intensity": 1},
	                                 )
	from scipy import interpolate


	def readdatafunc(
			filepath
	):
		R, X = readdatafunc0(filepath)
		R = np.squeeze(R)
		f = interpolate.interp1d(X, R, kind = "cubic")
		newX = np.linspace(400, 1800, 512)
		newR = f(newX)
		newR = np.expand_dims(newR, axis = 0)

		return newR, newX


	db_cfg = dict(
		dataroot = os.path.join(projectroot, "data", "liver_cell_dou"),

		LoadCsvFile = readdatafunc,
		backEnd = ".csv", t_v_t = [1., 0., 0.], newfile = False, k_split = k_split,
		transform = Process.process_series([  # 设置预处理流程
			Process.baseline_als(),
			# Process.bg_removal_niter_fit(),
			# Process.bg_removal_niter_piecewisefit(),
			Process.sg_filter(),
			Process.norm_func(), ]
		)
	)
	# transform = None, )

	recorddir = "Record_" + time.strftime("%Y-%m-%d-%H_%M_%S")
	recorddir = os.path.join(projectroot, "results", "liver_ml", recorddir)  # 实验结果保存位置
	for model in [ preprocess_LDA]:
		recordsubdir = os.path.join(recorddir,
		                            "Record" + time.asctime().replace(":", "-").replace(" ", "_"))
		if not os.path.isdir(recordsubdir):
			os.makedirs(recordsubdir)
		recordfile = recordsubdir + ".csv"
		f = open(recordfile, "w", newline = "")
		writer = csv.writer(f)
		f.write(db_cfg.__str__() + "\n")
		writer.writerow(["n", "k", "test_acc", "test_auc"])
		conf_m = None
		testaccs = []
		c = 0
		aucs = []
		for n in range(1):
			for k in range(k_split):
				sfpath = "Raman_" + str(n) + ".csv"
				train_db = raman(**db_cfg, mode = "train", k = k, sfpath = sfpath)
				val_db = raman(**db_cfg, mode = "val", k = k, sfpath = sfpath)
				num_classes = train_db.numclasses
				if conf_m is None:
					conf_m = np.zeros((train_db.numclasses, train_db.numclasses))
				if not n and not k:
					train_db.show_data(win = "train")
					val_db.show_data(win = "test")
				l1 = train_db.RamanFiles
				l2 = val_db.RamanFiles
				l = list(set(l1) & set(l2))
				print(len(l) / len(val_db))
				train_data, train_label = [np.squeeze(x.numpy()) for x in train_db.Ramans], [x.item() for x in
				                                                                             train_db.labels]
				test_data, test_label = [np.squeeze(x.numpy()) for x in val_db.Ramans], [x.item() for x in
				                                                                         val_db.labels]
				if model is not None:
					train_data, train_label, test_data, test_label = \
						model(train_data, train_label, test_data, test_label)
				classifier = svm.SVC(C = 2, kernel = 'rbf', gamma = 10, decision_function_shape = 'ovr',
				                     probability = True)  # ovr:一对多策略
				classifier.fit(train_data, train_label)

				pred = classifier.predict_proba(test_data)
				test_acc = classifier.score(test_data, test_label)
				testaccs.append(test_acc)

				label2auc = {}
				conf_m += confusion_matrix(test_label, classifier.predict(test_data))
				for i in range(num_classes):
					l_t = np.equal(test_label, i).astype(int)
					score = pred[:, i]
					frp, tpr, thresholds = roc_curve(l_t, score)
					plt.plot(frp, tpr)
					# label2auc[i] = auc(frp, tpr)
					label2auc[i] = roc_auc_score(l_t, score)
				auc_m = np.mean(list(label2auc.values()))
				aucs.append(auc_m)
				writer.writerow([n, k, test_acc, auc_m])
				c += 1
				print(c, "/", 1 * k_split)

		np.savetxt(os.path.join(recordsubdir, "test_confusion_matrix.csv"), conf_m, delimiter = ",")
		heatmap(conf_m, os.path.join(recordsubdir, "test_confusion_matrix.png"))
		ta = np.mean(np.array(testaccs)).__str__() + "+-" + np.std(np.array(testaccs)).__str__()
		auca = np.mean(np.array(testaccs)).__str__() + "+-" + np.std(np.array(testaccs)).__str__()
		writer.writerow(["mean", "std", ta, auca])
		if num_classes == 2:
			A, B, C, D = conf_m.flatten()
			acc = (A + D) / (A + B + C + D)
			sens = A / (A + B)
			spec = D / (C + D)
			f.write(
				"accuracy,{}\nsensitivity,{}\nspecificity,{}".format(acc, sens, spec)
			)
