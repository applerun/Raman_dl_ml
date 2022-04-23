from bacteria.code_sjh.utils.RamanData import Raman, projectroot
from bacteria.code_sjh.models import AlexNet_Sun
from bacteria.code_sjh.models.CNN.ResNet import ResNet18,ResNet34,ResNet50,ResNet101,ResNet152
from bacteria.code_sjh.utils.Validation import evaluate, evaluate_labelwise,startVisdomServer
from bacteria.code_sjh.utils.iterator import train
from bacteria.code_sjh.utils.Validation.visdom_utils import batch_plt
from bacteria.code_sjh.utils.Classifier import copy_filewise_classify
import torch, numpy, visdom
import csv, os, sys

from torch import nn, optim
from torch.utils.data import DataLoader

coderoot = os.path.dirname(os.path.dirname(__file__))


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


def AST_main(
		sample_tensor,
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
		model = AlexNet_Sun,
		# 选择分类模型
		epo_interv = 2,
		# 每隔epo_interv 验证一次
		# prog_bar = False,  # 实现进度条功能
		criteon = nn.CrossEntropyLoss(),
		# 选择criteon
		lr_decay_rate = 0.5,
		lr_decay_period = 60,
):
	alexnet = model(sample_tensor = sample_tensor, num_classes = test_db.num_classes()).to(device)  # 生成模型

	if not modelname is None:
		alexnet.set_model_name(modelname)

	optimizer = optim.Adam(alexnet.parameters(), lr = lr)

	# create loaders\
	train_loader = DataLoader(train_db, batch_size = batchsz, shuffle = True, num_workers = numworkers,
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
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	save_dir = os.path.join(save_dir, alexnet.model_name + ".mdl")

	if not os.path.exists(save_dir):
		with open(save_dir, "w", newline = ""):
			pass

	print("start training")

	for epoch in range(epochs):
		loss = train(alexnet, lr, device, train_loader, criteon, optimizer, epoch, mixed = False,
		             lr_decay_rate = lr_decay_rate,
		             lr_decay_period = lr_decay_period)

		vis.line([loss.item()], [global_step], win = "loss_" + str(k), update = "append")
		global_step += 1
		if epoch % epo_interv == 0 or epoch == epochs - 1:
			alexnet.eval()
			val_acc = evaluate(alexnet, val_loader, device)
			train_acc = evaluate(alexnet, train_loader, device)
			test_acc = evaluate(alexnet, test_loader, device)

			vis.line([val_acc], [global_step], win = "val_acc_" + str(k), update = "append")
			vis.line([train_acc], [global_step], win = "train_acc_" + str(k), update = "append")
			vis.line([test_acc], [global_step], win = "test_acc_" + str(k), update = "append")

			batch_plt(evaluate_labelwise(alexnet, val_db, device), global_step, win = "val_acc_each_sample" + str(k),
			          update = None if global_step <= epo_interv else "append", viz = vis)

			if val_acc >= best_acc:
				best_epoch = epoch
				best_acc = val_acc
				alexnet.save(save_dir)

	alexnet.load(save_dir)

	print("best_acc:", best_acc, "best epoch", best_epoch)
	print("loaded from ckpt!")
	test_acc = evaluate(alexnet, test_loader, device)
	copy_filewise_classify(test_db, alexnet,
	                       os.path.join(projectroot, "results", "radar", "test", alexnet.model_name + str(k)),
	                       device = device)
	copy_filewise_classify(val_db, alexnet,
	                       os.path.join(projectroot, "results", "radar", "val", alexnet.model_name + str(k)),
	                       device = device)
	print("test_acc:", test_acc)

	return best_acc, test_acc, best_epoch


if __name__ == '__main__':
	startVisdomServer()
	dataroot = os.path.join(os.path.dirname(coderoot), "data", "radar", "ver1")
	k_split = 4
	datasetcfg = dict(
		dataroot = dataroot,
		LoadCsvFile = radarfile2data,
		backEnd = ".csv",
		transform = None,
		t_v_t = [0.7, 0.2, 0.1],
		k_split = k_split,
		# ratio = {"Norm":0.5}
	)

	raman = Raman  # 数据集
	vis = visdom.Visdom()  # 监督
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 运算设备
	bestaccs, testaccs, bepochs = [], [], []  # 结果
	traincfg = dict(
		lr = 0.001,  # 选择learning rate
		epochs = 100,  # 选择epoch
		batchsz = 8,  # 选择batch size
		numworkers = 0,
		lr_decay_rate = 0.5,
		lr_decay_period = 60,
		device = device,
		vis = vis,
	)
	for model in [AlexNet_Sun,ResNet18,ResNet34,ResNet50,ResNet101,ResNet152]:
		for k in range(k_split):  # k-fold cross validation
			train_db = raman(**datasetcfg, mode = "train", k = k, newfile = True if k == 0 else False)
			val_db = raman(**datasetcfg, mode = "val", k = k)
			test_db = raman(**datasetcfg, mode = "test", k = k)
			sample_tensor, sample_label = train_db.__getitem__(1)
			sample_tensor = torch.unsqueeze(sample_tensor, dim = 0)
			b, t, be = AST_main(

				sample_tensor,
				train_db,
				val_db,
				test_db,
				model = model,
				**traincfg,
			)

			bestaccs.append(b)
			testaccs.append(t)
			bepochs.append(be)
		# train_db.shufflecsv()

		print("best acc:", numpy.mean(numpy.array(bestaccs)), "+-", numpy.std(numpy.array(bestaccs)))
		print("test acc", numpy.mean(numpy.array(testaccs)), "+-", numpy.std(numpy.array(testaccs)))

		print("best epochs", numpy.mean(numpy.array(bepochs)), "+-", numpy.std(numpy.array(bepochs)))