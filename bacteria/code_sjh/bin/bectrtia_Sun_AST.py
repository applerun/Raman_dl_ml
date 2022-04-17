import sys, os

import numpy

coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]

import torch, csv
import visdom
from torch import nn, optim

from bacteria.code_sjh.utils.RamanData import Raman, getRamanFromFile, Raman_dirwise
from bacteria.code_sjh.models.CNN.AlexNet import AlexNet_Sun
from bacteria.code_sjh.models.CNN.ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

from bacteria.code_sjh.utils.Validation.validation import *
from bacteria.code_sjh.utils.Validation.visdom_utils import *
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
		lr = 0.002,
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
	global_step = 0
	vis.line([0], [-1], win = "loss_" + str(k), opts = dict(title = "loss_" + str(k)))
	vis.line([0], [-1], win = "val_acc_" + str(k), opts = dict(title = "val_acc_" + str(k)))
	vis.line([0], [-1], win = "train_acc_" + str(k), opts = dict(title = "train_acc_" + str(k)))
	vis.line([0], [-1], win = "test_acc_" + str(k), opts = dict(title = "test_acc_" + str(k)))
	save_dir = os.path.join(save_dir, net.model_name + ".mdl")

	if not os.path.exists(save_dir):
		with open(save_dir, "w", newline = ""):
			pass

	print("start training")
	for epoch in range(epochs):
		loss = train(net, lr, device, train_loader, criteon, optimizer, epoch, mixed = False)
		vis.line([loss.item()], [global_step], win = "loss_" + str(k), update = "append")
		global_step += 1
		if epoch % epo_interv == 0 or epoch == epochs - 1:
			net.eval()
			val_acc = evaluate(net, val_loader, device)
			train_acc = evaluate(net, train_loader, device)
			# test_acc = evaluate(net, test_loader, device)
			# t0 = time.time()
			sample2acc_train = evaluate_samplewise(net, val_db, device)

			# t1 = time.time()
			# print("acc_cal_time: ",t1-t0)
			vis.line([val_acc], [global_step], win = "val_acc_" + str(k), update = "append")
			vis.line([train_acc], [global_step], win = "train_acc_" + str(k), update = "append")
			# vis.line([test_acc], [global_step], win = "test_acc_" + str(k), update = "append")
			# t2 = time.time()
			# print("acc draw_time: ",t2-t1)
			if verbose:
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
			# t3 = time.time()
			# print("batch draw time: ",t3-t2)
			batch_plt(sample2acc_train, global_step, win = "val_acc_each_sample"+str(k),
			          update = None if global_step <= epo_interv else "append", viz = vis)
			if val_acc >= best_acc and epoch > 20:
				best_epoch = epoch
				best_acc = val_acc
				net.save(save_dir)

	net.load(save_dir)

	print("best_acc:", best_acc, "best epoch", best_epoch)
	print("loaded from ckpt!")
	test_acc = evaluate(net, test_loader, device) if len(test_db) else 0

	print("test_acc:", test_acc)
	# sample2acc_test = evaluate_samplewise(alexnet, val_db + test_db, device)
	# print(sample2acc_test)
	return best_acc, test_acc, best_epoch


if __name__ == '__main__':
	startVisdomServer()
	vis = visdom.Visdom()  # python -m visdom.server

	# create model TODO:更换此处代码以更换不同的模型
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("using device:", device.__str__())
	from bacteria.code_sjh.config.CNN.Alexnet import Alexnet_Sun

	tvt = [0.8, 0.2, 0.0]

	bestaccs, testaccs, bepochs = [], [], []

	# dataroot = os.path.join(projectroot, "data", "liver", "liver_all_samplewise")
	dataroot = os.path.join(projectroot,"data","liver_cell")
	backend = ".asc"
	delimeter = "\t"
	dataformat = {"Wavelength": 0, "Column": 2, "Intensity": 1}
	raman = Raman_dirwise

	k_split = 10

	# readdatafunc = getRamanFromFile(wavelengthstart = 400,wavelengthend = 1800,delimeter = delimeter,dataname2idx = dataformat)
	readdatafunc0 = getRamanFromFile(
		wavelengthstart = 39, wavelengthend = 1810, delimeter = delimeter,
		dataname2idx = dataformat
	)
	from scipy import interpolate


	def readdatafunc(
			filepath
	):
		R, X = readdatafunc0(filepath)
		R = numpy.squeeze(R)
		f = interpolate.interp1d(X, R, kind = "cubic")
		newX = numpy.linspace(400, 1800, 512)
		newR = f(newX)
		newR = numpy.expand_dims(newR,axis = 0)
		return newR, newX


	from bacteria.code_sjh.utils import Process

	transform = Process.process_series(
		[
			Process.baseline_als(),
			Process.sg_filter(),
			Process.norm_func(),
		]
	)
	# model = AlexNet_Sun,  # 选择分类模型
	model = ResNet18

	db_cfg = dict(
		dataroot = dataroot, backEnd = backend, t_v_t = tvt, LoadCsvFile = readdatafunc,
		k_split = k_split, transform = transform
	)
	train_cfg = dict(
		device = device,
		batchsz = 32,
		vis = vis,
		lr = 0.0001,
		epochs = 120,
		verbose = False, )
	# config = dict(dataroot = os.path.join(projectroot, "data", "data_AST"), backEnd = backend, t_v_t = tvt, LoadCsvFile = getRamanFromFile(wavelengthstart = 0, wavelengthend = 1800, delimeter = delimeter,
	#                                  dataname2idx = dataformat), k_split = k_split)
	modellist = [AlexNet_Sun,ResNet18,ResNet34]
	n_iter = 1
	i = 0
	for model in modellist:
		recordfile = "ResNetRecord" + time.asctime().replace(":", "-").replace(" ", "_") + ".csv"
		f = open(recordfile, "w", newline = "")
		writer = csv.writer(f)
		f.write(db_cfg.__str__() + "\n")
		f.write(train_cfg.__str__() + "\n")
		writer.writerow(["n", "k", "best_acc", "test_acc", "best_epoch"])
		for n in range(1):
			for k in range(k_split):
				sfpath = "Raman_" + str(n) + ".csv"
				train_db = raman(**db_cfg, mode = "train", k = k, sfpath = sfpath)
				val_db = raman(**db_cfg, mode = "val", k = k, sfpath = sfpath)
				test_db = raman(**db_cfg, mode = "test", k = k, sfpath = sfpath)
				# train_db.show_data()
				sample_tensor, sample_label = train_db.__getitem__(1)
				vis.line(sample_tensor, win = "sampletensor")

				sample_tensor = torch.unsqueeze(sample_tensor, dim = 0)

				net = model(sample_tensor, test_db.num_classes()).to(device)
				b, t, be = AST_main(
					net,
					train_db,
					val_db,
					test_db,
					**train_cfg

				)
				writer.writerow([n, k, b, t, be])
				bestaccs.append(b)
				testaccs.append(t)
				bepochs.append(be)
				print(i,"/",len(modellist)*n_iter*k_split)
		# train_db.shufflecsv()
		ba = numpy.mean(numpy.array(bestaccs)).__str__() + "+-" + numpy.std(numpy.array(bestaccs)).__str__()
		ta = numpy.mean(numpy.array(testaccs)).__str__() + "+-" + numpy.std(numpy.array(testaccs)).__str__()
		bea = numpy.mean(numpy.array(bepochs)).__str__() + "+-" + numpy.std(numpy.array(bepochs)).__str__()
		writer.writerow(["mean", "std", ba, ta, bea])
		f.write("\n" + net.model_name)
		f.close()
		print("best acc:", ba)
		print("test acc", ta)
		print("best epochs", bea)
