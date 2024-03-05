import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn
from bacteria.code_sjh.Core.RamanData import RamanDatasetCore
from bacteria.code_sjh.Core.basic_functions.mpl_func import spectrum_vis_mpl

def plt_loss_acc(
		pltdir,
		res,
		informations = None):
	trainfig, trainax = plt.subplots(1, 2)  # 绘制训练过程图
	trainfig.suptitle('train process' + informations)

	epochs = np.arange(len(res["train_losses"]))

	loss_ax, acc_ax = trainax

	loss_ax.set_title("loss")
	loss_ax.plot(epochs, res["train_losses"], label = "train_loss", color = "red")
	loss_ax.plot(epochs, res["val_losses"], label = "val_loss", color = "blue")
	loss_ax.set_xlabel("epoch")
	loss_ax.set_ylabel("loss")
	loss_ax.legend()

	acc_ax.set_title("accuracy")
	acc_ax.plot(epochs, res["train_acces"], label = "train_accuracy", color = "red")
	acc_ax.plot(epochs, res["val_acces"], label = "val_accuracy", color = "blue")
	acc_ax.set_xlabel("epoch")
	acc_ax.set_ylabel("accuracy")
	acc_ax.legend()
	plt.subplots_adjust(wspace = 0.25)
	trainfig.savefig(os.path.join(pltdir, "train_process.png"))
	plt.close(trainfig)


def saveROC(dst, fpr, tpr, thretholds,auc):
	res = np.vstack((fpr, tpr, thretholds)).T
	np.savetxt(dst, res, delimiter = ",", header = "fpr,tpr,thretholds,AUC={}".format(auc))


def plt_res_val(pltdir,
				res,
				label2name,
				informations = None):
	if informations is None:
		informations = ""
	elif len(informations) > 0:
		informations = informations + "_"
	label2roc = res["label2roc"]
	label2auc = res["label2auc"]

	# ROC
	if not os.path.isdir(os.path.join(pltdir,"roc")):
		os.makedirs(os.path.join(os.path.join(pltdir,"roc")))
	for label in label2roc.keys():
		fpr, tpr, thresholds = label2roc[label]
		auc = label2auc[label]
		roc_fig, roc_ax = plt.subplots()
		roc_fig.suptitle("ROC_curve")
		roc_ax.set_title("auc = {}".format(auc))
		roc_ax.plot(fpr, tpr)
		roc_ax.set_xlabel("fpr")
		roc_ax.set_ylabel("tpr")
		roc_fig.savefig(os.path.join(pltdir, "roc", informations + label2name[label] + "_roc.png"))
		saveROC(os.path.join(pltdir, "roc", informations + label2name[label] + "_roc.csv"), fpr, tpr, thresholds,auc)
		plt.close(roc_fig)
	confusion_matrix = res["confusion_matrix"]
	cm_fig, cm_ax = plt.subplots()
	seaborn.heatmap(confusion_matrix, annot = True, cmap = "Blues", ax = cm_ax)
	cm_ax.set_title('confusion matrix')
	cm_ax.set_xlabel('predict')
	cm_ax.set_ylabel('true')
	cm_fig.savefig(os.path.join(pltdir, informations + "_confusion_matrix.png"))
	plt.close(cm_fig)


# def replt_cam(pltdir,db: RamanDatasetCore):
# 	camdir = os.path.join(pltdir,"cam")
# 	name2label = db.name2label()
# 	label2data = db.get_data_sorted_by_label()
# 	for file in os.listdir(camdir):
# 		if not file.endswith(".csv"):
# 			continue
# 		file_p = os.path.join(camdir,file)
# 		file_n = file[:-4]
# 		information,_,name = file.split("_")[0:3]
# 		label =name2label[name]

def plt_cam(pltdir,
			cams,
			db: RamanDatasetCore,
			informations = None):
	if informations is None:
		informations = ""
	label2name = db.label2name()
	label2data = db.get_data_sorted_by_label()
	fig, ax = plt.subplots(db.numclasses, 1)
	fig.suptitle(informations + "_grad_cam")
	for label in cams.keys():
		cam_ax = ax[label]
		cam = cams[label]
		if cam is None:
			continue
		name = label2name[label]
		data = label2data[label]
		spectrum_vis_mpl(data, xs = db.xs, name = 'spectrum_' + name, ax = cam_ax, line_color = "blue",
						 shadow_color = "skyblue")
		spectrum_vis_mpl(cam, xs = np.linspace(db.xs[0], db.xs[-1], cam.shape[-1]), name = "cam_" + name, ax = cam_ax,
						 line_color = "red", shadow_color = "pink")
		cam_ax.legend()
	plt.subplots_adjust(wspace = 0.25)
	fig.savefig(os.path.join(pltdir, informations + "_grad_cam"))
	plt.close(fig)


def plt_res(pltdir,
			res,
			val_db,
			test_db,
			informations = None):
	if informations is None:
		informations = ""
	label2name = val_db.label2name()

	plt_loss_acc(pltdir, res, informations)
	plt_res_val(pltdir, res["res_val"], label2name, informations = "val")
	plt_res_val(pltdir, res["res_test"], label2name, informations = "test")
	plt_cam(pltdir, res["val_cam"], val_db, "val")
	plt_cam(pltdir, res["test_cam"], test_db, "val")

def heatmap(matrix,
			path):
	cm_fig, cm_ax = plt.subplots()
	seaborn.heatmap(matrix, annot = True, cmap = "Blues", ax = cm_ax)
	cm_ax.set_title('confusion matrix')
	cm_ax.set_xlabel('predict')
	cm_ax.set_ylabel('true')
	cm_fig.savefig(path)
	plt.close(cm_fig)


def npsv(pltdir,
		 res,
		 val_db,
		 test_db,
		 ):
	process = np.array([res["train_acces"], res["val_acces"], res["train_losses"], res["val_losses"]]).T
	np.savetxt(os.path.join(pltdir, "train_process.csv"), process,
			   header = "train_acces,val_acces,train_losses,val_losses", delimiter = ",")
	os.makedirs(os.path.join(pltdir, "cam"))

	for label in test_db.label2name().keys():
		name = test_db.label2name()[label]
		try:
			val_cam = res["val_cam"][label]
			test_cam = res["test_cam"][label]
			xs = np.linspace(val_db.xs[0], val_db.xs[-1], val_cam.shape[-1])
			xs = np.expand_dims(xs, axis = 0)

			np.savetxt(os.path.join(pltdir, "cam", "val_cam_" + name + "_activated.csv"), np.vstack((xs, val_cam)),
					   delimiter = ",")
			np.savetxt(os.path.join(pltdir, "cam", "test_cam_" + name + "_activated.csv"), np.vstack((xs, test_cam)),
					   delimiter = ",")
		except:
			warnings.warn("cam output failed")

	return