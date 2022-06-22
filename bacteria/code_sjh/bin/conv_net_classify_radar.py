import sys, os
import time

import numpy
import numpy as np
import random, warnings, glob

coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
import torch, csv, seaborn
from torch import nn, optim
from bacteria.code_sjh.utils.Classifier import copy_filewise_classify, cam_output_filewise
from bacteria.code_sjh.models.CNN.AlexNet import AlexNet_Sun
from bacteria.code_sjh.models.CNN.ResNet import ResNet18, ResNet34
from bacteria.code_sjh.utils.Validation.validation import *
from bacteria.code_sjh.utils.Validation.mpl_utils import *
from bacteria.code_sjh.utils.Process_utils.errhandler import all_eval_err_handle
from bacteria.code_sjh.utils.iterator import train

from torch.utils.data import DataLoader

global loss
torch.backends.cudnn.benchmark = True


# 导入数据


def AST_main(
		net,
		# TODO:自动收集sample tensor
		train_db,
		val_db,
		test_db,
		# TODO:选择分配比例和数据文件夹
		device,
		# 选择训练键
		lr = 0.0002,
		# 选择learning rate
		epochs = 100,
		# 选择epoch
		batchsz = 40,
		# 选择batch size
		numworkers = 0,
		modelname = None,
		# 选择保存名
		vis = None,
		save_dir = os.path.join(coderoot, "checkpoints", ),
		# 选择保存目录
		epo_interv = 2,
		# 每隔epo_interv 验证一次
		# prog_bar = False,  # 实现进度条功能
		criteon = nn.CrossEntropyLoss(),
		# 选择criteon
		verbose = False,
		ax = None,
		lr_decay_rate = 0.5,
		lr_decay_period = 60,
):
	if vis == None:
		vis = visdom.Visdom()

	if not modelname is None:
		net.set_model_name(modelname)

	optimizer = optim.Adam(net.parameters(), lr = lr)

	# create loaders\
	train_loader = DataLoader(
		train_db, batch_size = batchsz, shuffle = True, num_workers = numworkers,
	)
	val_loader = DataLoader(val_db, batch_size = batchsz, num_workers = numworkers)
	test_loader = DataLoader(test_db, batch_size = batchsz, num_workers = numworkers)
	print("data loaded")
	# optimizer
	best_acc, best_epoch = 0, 0
	best_loss = -1
	global_step = 0

	steps = []
	train_losses, val_losses, test_losses = [], [], []
	train_acces, val_acces, test_accses = [], [], []

	vis.line([0], [-1], win = "loss_" + str(k), opts = dict(title = "loss_" + str(k)))
	vis.line([0], [-1], win = "val_acc_" + str(k), opts = dict(title = "val_acc_" + str(k)))
	vis.line([0], [-1], win = "train_acc_" + str(k), opts = dict(title = "train_acc_" + str(k)))
	# vis.line([0], [-1], win = "test_acc_" + str(k), opts = dict(title = "test_acc_" + str(k)))
	save_dir = os.path.join(save_dir, net.model_name + ".mdl")

	if not os.path.exists(save_dir):
		with open(save_dir, "w", newline = ""):
			pass

	print("start training")
	for epoch in range(epochs):

		loss = train(net, lr, device, train_loader, criteon, optimizer, epoch, mixed = False)
		vis.line([loss.item()], [global_step], win = "loss_" + str(k), update = "append")
		global_step += 1

		net.eval()
		with torch.no_grad():
			val_loss = evaluate_loss(net, val_loader, criteon, device)
			# test_loss = evaluate_loss(net,test_loader,criteon,device)
			train_acc = evaluate(net, train_loader, device)
			val_acc = evaluate(net, val_loader, device)
			# test_acc = evaluate(net, test_loader, device)

			train_acces.append(train_acc)
			val_acces.append(val_acc)
			# test_accses.append(test_acc)
			train_losses.append(loss)
			val_losses.append(val_loss)
			# test_losses.append(test_loss)
			steps.append(global_step)

			vis.line([val_acc], [global_step], win = "val_acc_" + str(k), update = "append")
			vis.line([train_acc], [global_step], win = "train_acc_" + str(k), update = "append")
			# vis.line([test_acc], [global_step], win = "test_acc_" + str(k), update = "append")
			# if val_acc >= best_acc and epoch > 20:
			if best_loss < 0 or val_loss < best_loss:
				best_epoch = epoch
				best_acc = val_acc
				net.save(save_dir)

		if epoch % epo_interv == 0 or epoch == epochs - 1:
			if verbose:
				sample2acc_train = evaluate_samplewise(net, val_db, device)
				sample2acc_val = evaluate_samplewise(net, val_db, device)
				sample2acc_test = evaluate_samplewise(net, test_db, device)
				batch_plt(
					sample2acc_val, global_step, win = "val_acc_each_sample" + str(k),
					update = None if global_step <= epo_interv else "append", viz = vis
				)
				batch_plt(
					sample2acc_test, global_step, win = "test_acc_each_sample" + str(k),
					update = None if global_step <= epo_interv else "append", viz = vis
				)
				batch_plt(sample2acc_train, global_step, win = "val_acc_each_sample" + str(k),
				          update = None if global_step <= epo_interv else "append", viz = vis)

	net.load(save_dir)

	# print("best_acc:", best_acc, "best epoch", best_epoch)
	# print("loaded from ckpt!")
	res_test = evaluate_all(net, test_loader, criteon, device)
	res_val = evaluate_all(net, val_loader, criteon, device)
	res = dict(train_acces = train_acces, val_acces = val_acces,  # 训练过程——正确率
	           train_losses = train_losses, val_losses = val_losses,  # 训练过程——损失函数
	           best_acc = best_acc, best_epoch = best_epoch,  # early-stopping位置
	           res_test = res_test,  # 测试集所有指标：
	           # acc：float正确率, loss:float,
	           # label2roc:dict 各个label的ROC, label2auc:dict 各个label的AUC, confusion_matrix:np.ndarray 混淆矩阵
	           res_val = res_val,  # 验证集所有指标

	           )

	label2data = val_db.get_data_sorted_by_label()
	cams = {}
	for l in label2data.keys():
		data = label2data[l]
		if data is None:
			continue
		data = data.to(device)
		test_cam = grad_cam(net, data, l, device)
		cams[l] = test_cam
	res["val_cam"] = cams
	label2data = test_db.get_data_sorted_by_label()
	cams = {}
	for l in label2data.keys():
		data = label2data[l]
		if data is None:
			cams[l] = None
			continue
		data = data.to(device)
		test_cam = grad_cam(net, data, l, device)
		cams[l] = test_cam
	res["test_cam"] = cams

	# sample2acc_test = evaluate_samplewise(alexnet, val_db + test_db, device)
	# print(sample2acc_test)

	return res


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


def plt_res_val(pltdir,
                res,
                label2name,
                informations = None,
                ticks = None):
	for s_dirs in ["roc", "conf_matrix"]:
		if not os.path.isdir(os.path.join(pltdir, s_dirs)):
			os.makedirs(os.path.join(pltdir, s_dirs))
	if ticks is None:
		ticks = "auto"
	if informations is None:
		informations = ""
	label2roc = res["label2roc"]
	label2auc = res["label2auc"]
	for label in label2roc.keys():
		fpr, tpr, thresholds = label2roc[label]
		auc = label2auc[label]
		roc_fig, roc_ax = plt.subplots()
		roc_fig.suptitle("ROC_curve")
		roc_ax.set_title("auc = {}".format(auc))
		roc_ax.plot(fpr, tpr)
		roc_ax.set_xlabel("fpr")
		roc_ax.set_ylabel("tpr")

		roc_fig.savefig(os.path.join(pltdir, "roc", informations + "_" + label2name[label] + "_roc.png"))
		plt.close(roc_fig)
	confusion_matrix = res["confusion_matrix"]
	cm_fig, cm_ax = plt.subplots()
	seaborn.heatmap(confusion_matrix, annot = True, cmap = "Blues", ax = cm_ax,
	                xticklabels = ticks, yticklabels = ticks)
	cm_ax.set_title('confusion matrix')
	cm_ax.set_xlabel('predict')
	cm_ax.set_ylabel('true')
	cm_fig.savefig(os.path.join(pltdir, "conf_matrix", informations + "_confusion_matrix.png"))
	np.savetxt(os.path.join(pltdir, "conf_matrix", informations + "_confusion_matrix.csv"), confusion_matrix,
	           delimiter = ",")
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
	if not os.path.isdir(os.path.join(pltdir, "cam")):
		os.makedirs(os.path.join(pltdir, "cam"))
	if informations is None:
		informations = ""
	label2name = db.label2name()
	fig, ax = plt.subplots(1, db.numclasses)
	fig.suptitle(informations + "_grad_cam")
	for label in cams.keys():
		cam_ax = ax[label]
		cam = cams[label]
		if cam is None:
			continue
		name = label2name[label]
		spectrum_vis_mpl(cam, xs = np.linspace(db.xs[0], db.xs[-1], cam.shape[-1]), name = "cam_" + name, ax = cam_ax,
		                 line_color = "red", shadow_color = "pink")
		cam_ax.legend()
	plt.subplots_adjust(wspace = 0.25)
	fig.savefig(os.path.join(pltdir, "cam", informations + "_grad_cam"))
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
	plt_res_val(pltdir, res["res_val"], label2name, informations = "val", ticks = val_db.label2name().values())
	plt_res_val(pltdir, res["res_test"], label2name, informations = "test", ticks = test_db.label2name().values())
	plt_cam(pltdir, res["val_cam"], val_db, "val")
	plt_cam(pltdir, res["test_cam"], test_db, "val")


# res = dict(train_acces = train_acces, val_acces = val_acces,  # 训练过程——正确率
#            train_losses = train_losses, val_losses = val_losses,  # 训练过程——损失函数
#            best_acc = best_acc, best_epoch = best_epoch,  # early-stopping位置
#            res_test = res_test,  # 测试集所有指标：
#            # acc：float正确率, loss:float,
#            # label2roc:dict 各个label的ROC, label2auc:dict 各个label的AUC, confusion_matrix:np.ndarray 混淆矩阵
#            res_val = res_val,  # 验证集所有指标
#            cams = cams,  # 梯度加权类激活映射图谱
#            )
def heatmap(matrix,
            path,
            ticks = None):
	if ticks is None:
		ticks = "auto"
	cm_fig, cm_ax = plt.subplots()
	seaborn.heatmap(matrix, annot = True, cmap = "Blues", ax = cm_ax, xticklabels = ticks, yticklabels = ticks)
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
	if not os.path.isdir(os.path.join(pltdir, "cam")):
		os.makedirs(os.path.join(pltdir, "cam"))
	for label in test_db.label2name().keys():
		name = test_db.label2name()[label]
		val_cam = res["val_cam"][label]
		test_cam = res["test_cam"][label]
		xs = np.linspace(val_db.xs[0], val_db.xs[-1], val_cam.shape[-1])
		xs = np.expand_dims(xs, axis = 0)

		np.savetxt(os.path.join(pltdir, "cam", "val_cam_" + name + "_activated.csv"), np.vstack((xs, val_cam)),
		           delimiter = ",")
		np.savetxt(os.path.join(pltdir, "cam", "test_cam_" + name + "_activated.csv"), np.vstack((xs, test_cam)),
		           delimiter = ",")
	return


def radarfile2data(filename):
	res = None
	with open(filename, "r", newline = "") as f:
		reader = csv.reader(f)
		for line in reader:
			t = numpy.array([float(x) for x in line[0:1260]])
			t = numpy.expand_dims(t, axis = 0)
			if res is None:
				res = t
			else:
				res = numpy.vstack((res, t))
	x = numpy.linspace(0, 26, res.shape[-1])
	return res, x


class radarData(Raman):
	def __init__(self,
	             *args,
	             **kwargs):
		"""

		:param dataroot: 数据的根目录
		:param resize: 光谱长度(未实装)
		:param mode: "train":训练集 "val":验证集 "test":测试集
		:param t_v_t:[float,float,float] 分割所有数据train-validation-test的比例
		:param savefilemode: 数据存储方式：1lamen1file:每个文件中有一个光谱，combinedfile:所有同label的光谱在一个文件中
		:param sfpath: 数据文件的名称，初始化时，会在数据根目录创建记录数据的csv文件，文件格式：label，*spectrum，如果已经有该记录文件
		:param shuffle: 是否将读取的数据打乱
		:param transform: 数据预处理/增强

		:param LoadCsvFile:function callabel 1lamen1file:根据数据存储格式自定义的读取文件数据的函数\
		combined:生成器，第一个为光谱数据的header
		:param backEnd:str 存储文件的后缀
		:param supervised: 如果为无监督学习，将noising前的信号设置为label
		:param noising: callable——input：1d spectrum output：noised 1d spectrum

		"""
		# assert mode in ["train", "val", "test"]
		if not "sfpath" in kwargs.keys():
			kwargs["sfpath"] = "labels.txt"
		super(radarData, self).__init__(*args, **kwargs)
		return

	def LoadCsv(self,
	            filename, ):
		header = ["label", "filepath"]
		if os.path.exists(os.path.join(self.root, filename)) and self.new:
			warnings.warn("old data file will be removed: {}".format(filename))
			assert filename != "labels.txt", "labels.txt is protected, please use another name."

		if not os.path.exists(os.path.join(self.root, filename)) or self.new:
			RamanFiles = []
			for name in self.name2label.keys():
				files = glob.glob(os.path.join(self.root, name, "*" + self.dataEnd))
				if self.ratio is not None:
					if not name in self.ratio.keys():
						ratio = 1.0
					else:
						ratio = self.ratio[name]
					if ratio < 1.0:
						files = random.sample(files, int(ratio * len(files)))
					elif ratio > 1.0:
						pass  # TODO:过采样函数

				RamanFiles += files

			if self.shuff:  # 打乱顺序
				random.shuffle(RamanFiles)
			with open(os.path.join(self.root, filename), mode = "w", newline = "") as f:  # 记录所有数据
				writer = csv.writer(f)

				writer.writerow(header)

				for spectrum in RamanFiles:  # spectrum:data root/label name/**.csv
					name = spectrum.split(os.sep)[-2]  # label name
					label = self.name2label[name]  # label idx

					writer.writerow([label, spectrum])

		self.RamanFiles = []
		self.labels = []

		with open(os.path.join(self.root, filename)) as f:
			reader = csv.reader(f)
			if not filename == "labels.txt":
				for row in reader:
					if row == header:
						continue
					try:
						label = int(row[0])
						spectrum = row[1]
						self.labels.append(torch.tensor(label))
						self.RamanFiles.append(spectrum)
					except:  # 数据格式有误
						print("wrong csv,remaking...")
						f.close()
						os.remove(os.path.join(self.root, filename))
						self.LoadCsv(filename)
						break
			else:
				l = list(reader)
				self.RamanFiles, labelnames = l[0], l[2]
				for i in range(len(labelnames)):
					if i == len(labelnames):
						break
					while not labelnames[i] in self.name2label.keys():
						del self.RamanFiles[i]
						del labelnames[i]
						if i == len(labelnames):
							break
					self.RamanFiles[i] = os.path.join(self.root, labelnames[i], self.RamanFiles[i])
					labelnames[i] = torch.tensor(self.name2label[labelnames[i]])
				self.labels = labelnames
		assert len(self.RamanFiles) == len(self.labels)
		return self.RamanFiles, self.labels


if __name__ == '__main__':
	startVisdomServer()  # python -m visdom.server    启动visdom本地服务器
	vis = visdom.Visdom()  # visdom对象

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运算设备
	print("using device:", device.__str__())

	# 记录训练结果
	bestaccs, testaccs, bepochs, vaucs, taucs = [], [], [], [], []

	raman = radarData  # 设置读取数据集的DataSet

	dataroot = os.path.join(os.path.dirname(coderoot), "data", "radar", "class3ver3")
	k_split = 8


	def readdata(filename):
		data, x = radarfile2data(filename)
		data = data[1, 900:1260]
		data = np.expand_dims(data, axis = 0)
		return data, x[900:1260]


	datasetcfg = dict(
		dataroot = dataroot,
		LoadCsvFile = readdata,
		backEnd = ".csv",
		transform = all_eval_err_handle,

		t_v_t = [0.7, 0.2, 0.1],
		# t_v_t = [1.,0.,0.],
		k_split = k_split,
		# ratio = {"Norm":0.5}
		# sfpath = "labels_shuffled.csv"
	)
	traincfg = dict(
		lr = 0.0001,  # 选择learning rate
		epochs = 100,  # 选择epoch
		batchsz = 16,  # 选择batch size
		numworkers = 0,
		lr_decay_rate = 0.5,
		lr_decay_period = 60,
		device = device,
		vis = vis,
	)
	modellist = [AlexNet_Sun]  # 需要验证的模型
	n_iter = 1  # 交叉验证重复次数
	i = 0  # 实验进度计数

	recorddir = "Record_" + time.strftime("%Y-%m-%d-%H_%M_%S")
	recorddir = os.path.join(projectroot, "results", "radar", recorddir)  # 实验结果保存位置

	if not os.path.isdir(recorddir):
		os.makedirs(recorddir)
	for model in modellist:
		recordsubdir = os.path.join(recorddir,
		                            "Record" + time.asctime().replace(":", "-").replace(" ", "_"))  # 每个模型一个文件夹保存结果
		if not os.path.isdir(recordsubdir):
			os.makedirs(recordsubdir)
		recordfile = recordsubdir + ".csv"  # 记录训练的配置和结果

		f = open(recordfile, "w", newline = "")
		writer = csv.writer(f)
		f.write(datasetcfg.__str__() + "\n")
		f.write(traincfg.__str__() + "\n")
		writer.writerow(["n", "k", "best_acc", "test_acc", "best_epoch", "val_AUC", "test_AUC"])
		conf_m_v = None
		conf_m_t = None
		for n in range(n_iter):
			for k in range(k_split):
				sfpath = "Raman_" + str(n) + ".csv"
				train_db = raman(**datasetcfg, mode = "train", k = k, sfpath = sfpath)
				val_db = raman(**datasetcfg, mode = "val", k = k, sfpath = sfpath)
				if conf_m_v is None:
					conf_m_v = np.zeros((train_db.numclasses, train_db.numclasses))
					conf_m_t = np.zeros((train_db.numclasses, train_db.numclasses))
				assert len(val_db) > 0, str(val_db.sfpath) + ":" + str(val_db.RamanFiles)
				test_db = raman(**datasetcfg, mode = "test", k = k, sfpath = sfpath)
				# train_db.show_data()
				sample_tensor, sample_label = train_db.__getitem__(1)
				vis.line(sample_tensor, win = "sampletensor")

				sample_tensor = torch.unsqueeze(sample_tensor, dim = 0)

				net = model(sample_tensor, test_db.num_classes()).to(device)
				res = AST_main(
					net,
					train_db,
					val_db,
					test_db,
					**traincfg,
				)

				pltdir = os.path.join(recordsubdir, "n-{}-k-{}".format(n, k))
				if not os.path.isdir(pltdir):
					os.makedirs(pltdir)
				b, t, be, auc_val, auc_test = res["best_acc"], res["res_test"]["acc"], res["best_epoch"], \
				                              np.mean(list(res["res_val"]["label2auc"].values())), \
				                              np.mean(list(res["res_test"]["label2auc"].values()))
				writer.writerow([n, k, b, t, be, auc_val, auc_test])
				bestaccs.append(b)
				testaccs.append(t)
				bepochs.append(be)
				vaucs.append(auc_val)
				taucs.append(auc_test)
				i += 1
				print(i, "/", len(modellist) * n_iter * k_split)
				plt_res(pltdir,
				        res,
				        val_db,
				        val_db,
				        informations = None)
				npsv(pltdir, res, val_db, val_db, )
				conf_m_v += res["res_val"]["confusion_matrix"]
				conf_m_t += res["res_test"]["confusion_matrix"]
				cam_output_filewise(val_db, net,
				                    os.path.join(projectroot, "results", "radar", recordsubdir, "val", "cam", ))
				copy_filewise_classify(test_db, net,
				                       os.path.join(projectroot, "results", "radar", recordsubdir, "test",
				                                    net.model_name + str(k)),
				                       device = device)
				copy_filewise_classify(val_db, net,
				                       os.path.join(projectroot, "results", "radar", recordsubdir, "val",
				                                    net.model_name + str(k)),
				                       device = device)
				net.save(
					os.path.join(projectroot, "results", "radar", recordsubdir, "n-{}-k-{}".format(n, k), "net.mdl"))
		np.savetxt(os.path.join(recordsubdir, "test_confusion_matrix.csv"), conf_m_v, delimiter = ",")
		np.savetxt(os.path.join(recordsubdir, "val_confusion_matrix.csv"), conf_m_t, delimiter = ",")
		heatmap(conf_m_t, os.path.join(recordsubdir, "test_confusion_matrix.png"), ticks = test_db.label2name().keys())
		heatmap(conf_m_v, os.path.join(recordsubdir, "val_confusion_matrix.png"), ticks = test_db.label2name().keys())
		# train_db.shufflecsv()
		ba = np.mean(numpy.array(bestaccs)).__str__() + "+-" + numpy.std(numpy.array(bestaccs)).__str__()
		ta = np.mean(numpy.array(testaccs)).__str__() + "+-" + numpy.std(numpy.array(testaccs)).__str__()
		bea = np.mean(numpy.array(bepochs)).__str__() + "+-" + numpy.std(numpy.array(bepochs)).__str__()
		auc_av = np.mean(numpy.array(vaucs)).__str__() + "+-" + numpy.std(numpy.array(vaucs)).__str__()
		auc_at = np.mean(numpy.array(taucs)).__str__() + "+-" + numpy.std(numpy.array(taucs)).__str__()

		writer.writerow(["mean", "std", ba, ta, bea, auc_av, auc_at])
		f.write("\n" + net.model_name)
		f.close()
		print("best acc:", ba)
		print("test acc", ta)
		print("best epochs", bea)
