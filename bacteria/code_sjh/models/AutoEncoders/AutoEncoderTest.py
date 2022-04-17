import csv

import torch
import visdom
from torch import optim
from torch.utils.data import DataLoader
# 记录各个路径
import os, sys

coderoot = os.path.join(*os.path.abspath(__file__).split(os.sep)[:-3]).replace(":", ":" + os.sep)

projectroot = os.path.split(coderoot)[0]
# projectroot = os.path.split(projectroot)[0]
dataroot = os.path.join(os.path.split(projectroot)[0], "data", "data_AST")

sys.path.append(coderoot)

from bacteria.code_sjh.models.Criteons.CrossEntropy import BCE
from bacteria.code_sjh.utils import RamanData, iterator, Process
from bacteria.code_sjh.utils import Validation as validation
from bacteria.code_sjh.utils.Process import noising_func_generator
from bacteria.code_sjh.models.BasicModule import BasicModule, Flat
from AutoEncoder import *
from bacteria.code_sjh.models.CNN.SVM import *
from bacteria.code_sjh.models.Parts import *
from VAE import *

PCA_flag = False

if not os.path.exists(os.path.join(coderoot, "checkpoints", "convAEencoders")):
	os.mkdir(os.path.join(coderoot, "checkpoints", "convAEencoders"))


def main_test(batchsz = 16,
              numworkers = 0,
              epochs = 150,
              lr = 0.0001,
              modelname = None,
              AutoEncoder = ConvAutoEncoder,
              neck_axis = 2):
	from sklearn.decomposition import PCA

	tvt = [0.8, 0.2, 0]
	series = [Process.baseline_als(lam = 100000, p = 0.01, niter = 10),
	          Process.sg_filter(window_length = 11, polyorder = 3),
	          Process.norm_func(a = 0, b = 1),
	          lambda x: x[0:524]
	          ]
	noising_func = noising_func_generator(0.05)
	# noising_func = None
	process = Process.process_series(series)
	savefilemode = "combined"

	train_db = RamanData.Raman(dataroot, mode = "train", backEnd = ".csv", t_v_t = tvt, savefilemode = savefilemode,
	                           transform = process,
	                           noising = noising_func, unsupervised = True)
	val_db = RamanData.Raman(dataroot, mode = "val", backEnd = "-.csv", t_v_t = tvt, savefilemode = savefilemode,
	                         transform = process)
	train_loader = DataLoader(train_db, batch_size = batchsz, shuffle = True, num_workers = numworkers)

	print("data loaded")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("using device:", device.__str__())

	vis = visdom.Visdom()

	sample_tensor, sample_label = train_db.__getitem__(1)
	xs = torch.linspace(400, 1800, sample_tensor.shape[-1])
	# vis.line(torch.squeeze(sample_label), xs, name = "sample_label", win = "sample", opts = dict(
	#     title = "sample"
	# ))
	# vis.line(torch.squeeze(sample_tensor), xs, name = "sample_tensor", win = "sample", update = "append", opts = dict(
	#     title = "sample"
	# ))

	sample_tensor = torch.unsqueeze(sample_tensor, dim = 0)
	model = AutoEncoder(sample_tensor, neck_axis = neck_axis).to(device)
	model.model_loaded = True

	if modelname:
		model.set_model_name(modelname)
	print(model)
	optimizer = optim.Adam(model.parameters(), lr = lr, )

	# criteon = nn.MSELoss()
	criteon = BCE()

	vis.line([0], [-1], win = "loss", opts = dict(title = "loss"))

	save_dir = os.path.join(coderoot, "checkpoints", model.model_name + ".mdl")
	if not os.path.exists(save_dir):
		with open(save_dir, "w", newline = ""):
			pass

	print("start training")

	label2data_val = val_db.get_data_sorted_by_label()
	label2data_train = train_db.get_data_sorted_by_label()

	label2name_train = train_db.label2name()
	label2name_val = val_db.label2name()
	for epoch in range(epochs):
		model.train()
		loss = iterator.train(model, lr, device, train_loader, criteon, optimizer, epoch)
		vis.line([loss.item()], [epoch], win = "loss", update = "append")
		if (epoch % 20 == 9 or epoch == epochs - 1) and epoch > 30:
			model.eval()
			with torch.no_grad():
				for idx in range(len(label2name_val.keys())):
					spectrum = label2data_val[idx]
					spectrum = spectrum.to(device)
					spectrum_hat = model(spectrum)
					win = "spectrum_AE_val_" + label2name_val[idx] + "_epoch:" + str(epoch + 1)
					validation.spectrum_vis(spectrum, xs, win, name = "x", vis = vis)
					validation.spectrum_vis(spectrum_hat, xs, win, update = "append", name = "x_hat", vis = vis)
					if neck_axis < 4:
						model.neck_vis(spectrum, label2name_val[idx], vis = vis,
						               update = None if idx == 0 else "append",
						               win = "val_neck_vis_epoch:" + str(epoch + 1))
				for idx in range(len(label2name_train.keys())):
					spectrum = label2data_train[idx]
					spectrum = spectrum.to(device)
					spectrum_hat = model(spectrum)
					win = "spectrum_AE_train_" + label2name_train[idx] + "_epoch:" + str(epoch + 1)
					validation.spectrum_vis(spectrum, xs, win, name = "x", vis = vis)
					validation.spectrum_vis(spectrum_hat, xs, win, update = "append", name = "x_hat", vis = vis)
					if neck_axis < 4:
						model.neck_vis(spectrum, label2name_train[idx], vis = vis,
						               update = None if idx == 0 else "append",
						               win = "train_neck_vis_epoch:" + str(epoch + 1))
				model.Encoder.save(os.path.join(coderoot, "checkpoints", "convAEencoders",
				                                "convAE_encoder_e" + str(epoch + 1) + ".mdl"))
	model.save(os.path.join(coderoot, "checkpoints", model.model_name + ".mdl"))

	model.eval()
	for idx in range(len(label2name_train.keys())):
		with torch.no_grad():
			spectrum = label2data_val[idx]
			spectrum = spectrum.to(device)
			spectrum_hat = model(spectrum)

		win = "spectrum_AE" + label2name_train[idx]

		validation.spectrum_vis(spectrum, xs, win, name = "x", vis = vis)
		validation.spectrum_vis(spectrum_hat, xs, win, update = "append", name = "x_hat", vis = vis)
		if neck_axis < 4:
			model.neck_vis(spectrum, label2name_train[idx], vis = vis, update = None if idx == 0 else "append")

		# 使用PCA作对比
		if PCA_flag is False:
			pca = PCA(n_components = 2)
			all_loader = DataLoader(train_db, batch_size = len(train_db), shuffle = True, num_workers = numworkers)
			train_data, _ = iter(all_loader).__next__()
			train_data = torch.squeeze(train_data)
			pca.fit(train_data.numpy())
			spectrum = label2data_val[idx]  # [b,c=1,l]
			spectrum = torch.squeeze(spectrum)  # [b,l]
			newX = pca.transform(spectrum.numpy())

			vis.scatter(newX,
			            win = "PCA",
			            update = "append" if idx > 0 else None,
			            name = label2name_train[idx],
			            opts = dict(
				            title = "PCA",
				            markersize = 6,
				            showlegend = True,
				            markersymbol = "cross",
			            )
			            )

	return False


def VAE_test(batchsz = 16, numworkers = 0, epochs = 1000, lr = 0.0005, modelname = "VAE", VAE = VAE_conv4,
             neck_axis = 2, save_dir = None, ):
	if save_dir is None:
		save_dir = os.path.join(coderoot, "checkpoints", "VAE")
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
		os.mkdir(os.path.join(save_dir, "trained"))
		os.mkdir(os.path.join(save_dir, "encoder"))
		pass

	tvt = [0.8, 0.2, 0]

	series = [Process.baseline_als(lam = 100000, p = 0.01, niter = 10),
	          Process.sg_filter(window_length = 11, polyorder = 3),
	          Process.norm_func(a = 0, b = 1),
	          lambda x: x[0:517]
	          ]

	# prepare for training
	# noising_func = noising_func_generator(0.05)
	noising_func = None
	process = Process.process_series(series)
	train_db = RamanData.Raman(dataroot, mode = "train", backEnd = "-.csv", t_v_t = tvt, transform = process,
	                           noising = noising_func, unsupervised = True)
	val_db = RamanData.Raman(dataroot, mode = "val", backEnd = "-.csv", t_v_t = tvt, transform = process)
	train_loader = DataLoader(train_db, batch_size = batchsz, shuffle = True, num_workers = numworkers)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	vis = visdom.Visdom()
	sample_tensor, sample_label = train_db.__getitem__(1)
	sample_tensor = torch.unsqueeze(sample_tensor, dim = 0)
	model = VAE(sample_tensor, neck_axis = neck_axis, dropout = 0.1).to(device)
	model.model_loaded = True
	print(model)  # the structure of the model
	xs = torch.linspace(400, 1800, sample_tensor.shape[-1])
	if modelname:
		model.set_model_name(modelname)
	optimizer = optim.Adam(model.parameters(), lr = lr, )

	# 设置loss function
	criteon = BCE()
	# criteon = nn.MSELoss()

	vis.line([0], [-1], win = "VAE_loss", opts = dict(title = "VAE_loss"))  # 监控训练loss

	# print("start training")
	label2data_val = val_db.get_data_sorted_by_label()
	label2data_train = train_db.get_data_sorted_by_label()
	label2name = val_db.label2name()
	model.train()
	for epoch in range(epochs):

		loss = iterator.train(model, lr, device, train_loader, criteon, optimizer, epoch,
		                      loss_plus = lambda x: 0.2 * x, lr_decay_rate = 0.8, lr_decay_period = 60)
		vis.line([loss.item()], [epoch], win = "VAE_loss", update = "append")
		if epoch % 30 == 9 or epoch == epochs - 1:
			model.eval()
			model.save(os.path.join(save_dir, "trained", "epoch{}.mdl".format(epoch + 1)))
			model.Encoder.save(os.path.join(save_dir, "encoder", "epoch{}.mdl".format(epoch + 1)))
			for idx in range(len(label2name.keys())):
				with torch.no_grad():
					spectrum = label2data_val[idx]
					spectrum = spectrum.to(device)
					spectrum_hat = model(spectrum)[0]
					win = "spectrum_VAE_val_" + label2name[idx] + "epoch:" + str(epoch + 1)
					validation.spectrum_vis(spectrum, xs, win, name = "x", vis = vis)
					validation.spectrum_vis(spectrum_hat, xs, win, update = "append", name = "x_hat", vis = vis)
					if neck_axis < 4:
						model.neck_vis(spectrum, label2name[idx], vis = vis, update = None if idx == 0 else "append",
						               win = "VAE_val_neck_vis_epoch:" + str(epoch + 1))

					spectrum = label2data_train[idx]
					spectrum = spectrum.to(device)
					spectrum_hat = model(spectrum)[0]
					win = "spectrum_VAE_train_" + label2name[idx] + "epoch:" + str(epoch + 1)
					validation.spectrum_vis(spectrum, xs, win, name = "x", vis = vis)
					validation.spectrum_vis(spectrum_hat, xs, win, update = "append", name = "x_hat", vis = vis)
					if neck_axis < 4:
						model.neck_vis(spectrum, label2name[idx], vis = vis, update = None if idx == 0 else "append",
						               win = "VAE_train_neck_vis_epoch:" + str(epoch + 1))
			model.train()
	model.eval()
	for idx in range(len(label2name.keys())):
		with torch.no_grad():
			spectrum = label2data_val[idx]
			spectrum = spectrum.to(device)
			spectrum_hat = model(spectrum)[0]
		win_main = "spectrum_VAE"
		win = win_main + label2name[idx]

		validation.spectrum_vis(spectrum, xs, win, name = "x", vis = vis)
		validation.spectrum_vis(spectrum_hat, xs, win, update = "append", name = "x_hat", vis = vis)
		if neck_axis < 4:
			model.neck_vis(spectrum, label2name[idx], win = win_main, vis = vis,
			               update = None if idx == 0 else "append")


def mae_test(savefilemode = "combined",
             batchsz = 16,
             numworkers = 0,
             mutedencoder = muted_VAEncoder,
             AutoEncoder = VAE_conv4,
             neck_axis = None,
             save_mdlfile = os.path.join(coderoot, "checkpoints_encoder.mdl")
             ):
	series = [Process.baseline_als(lam = 100000, p = 0.01, niter = 10),
	          Process.sg_filter(window_length = 11, polyorder = 3),
	          Process.norm_func(a = 0, b = 1),
	          lambda x: x[0:524]
	          ]
	print("series")
	tvt = [0.8, 0.2, 0]
	process = Process.process_series(series)
	train_db = RamanData.Raman(dataroot, mode = "train", backEnd = ".csv", t_v_t = tvt, savefilemode = savefilemode,
	                           transform = process,
	                           unsupervised = False)
	# print("train_db")
	cae = AutoEncoder(torch.unsqueeze(train_db.__getitem__(0)[0], dim = 0), neck_axis = neck_axis)
	me = mutedencoder(cae, save_mdlfile)
	# print("me")
	# 新的训练集
	series.append(me)
	process = Process.process_series(series)
	train_db = RamanData.Raman(dataroot, mode = "train", backEnd = ".csv", t_v_t = tvt, savefilemode = savefilemode,
	                           transform = process,
	                           unsupervised = False)
	val_db = RamanData.Raman(dataroot, mode = "val", backEnd = ".csv", t_v_t = tvt, savefilemode = savefilemode,
	                         transform = process,
	                         unsupervised = True)
	label2data_val = val_db.get_data_sorted_by_label()
	vis = visdom.Visdom()

	if neck_axis < 4:
		win = "unclassified data"
		for label in label2data_val.keys():
			data = torch.squeeze(label2data_val[label])
			vis.scatter(data,
			            win = win,
			            update = None if label is 0 else "append",
			            name = str(label),
			            opts = dict(
				            title = win,
				            showlegend = True,
			            )
			            )

	# train_loader = DataLoader(train_db, batch_size = batchsz, shuffle = True, num_workers = numworkers)
	# print("data loaded")
	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# print("using device:", device.__str__())

	# 使用SVM分类
	from sklearn import svm
	mlmodel = svm.SVC()
	X_train = torch.squeeze(torch.stack(train_db.Ramans, dim = 0)).numpy()
	y_train = torch.squeeze(torch.stack(train_db.class_num, dim = 0)).numpy()

	mlmodel.fit(X = X_train, y = y_train)

	res1 = mlmodel.score(y = y_train, X = X_train)

	X_val = torch.squeeze(torch.stack(val_db.Ramans, dim = 0)).numpy()
	y_val = torch.squeeze(torch.stack(val_db.class_num, dim = 0)).numpy()
	res2 = mlmodel.score(y = y_val, X = X_val)

	return res1, res2


if __name__ == '__main__':
	validation.startVisdomServer()
	# train_accs = []
	# test_accs = []
	# necks = []
	# epochs = []
	# mdlroot = os.path.join(coderoot, "checkpoints", "convAEencoders")
	# with open("record.csv", "w+") as f:
	# 	lenthOfData = len(f.readlines())
	# 	w = csv.writer(f)
	#
	# 	for a in range(4, 12, 2):
	# 		if len(os.listdir(mdlroot)) > 0 and lenthOfData % len(os.listdir(mdlroot)) > 0:
	# 			lenthOfData -= len(os.listdir(mdlroot))
	# 			continue
	# 		main_test(epochs = 100, neck_axis = a)
	# 		PCA_flag = True
	# 		for file in os.listdir(mdlroot):
	# 			if lenthOfData > 0:
	# 				lenthOfData -= 1
	# 				continue
	# 			e = int("".join(filter(str.isdigit, file)))
	# 			train_acc, test_acc = mae_test(neck_axis = a,
	# 			                               save_mdlfile = os.path.join(coderoot, "checkpoints", "convAEencoders",
	# 			                                                           file))
	# 			train_accs.append(train_acc)
	# 			epochs.append(e)
	# 			necks.append(a)
	# 			test_accs.append(test_acc)
	# 			w.writerow([e, a, train_acc, test_acc])
	# vis = visdom.Visdom()
	# train_accs = torch.tensor(train_accs)
	# test_accs = torch.tensor(test_accs)
	# necks = torch.tensor(necks)
	# epochs = torch.tensor(epochs)
	# vis.scatter(torch.stack((necks, epochs, train_accs,), dim = 0).T, win = "train_acc")
	# vis.scatter(torch.stack((necks, epochs, test_accs,), dim = 0).T, win = "test_acc")

	# VAE_test(epochs = 400)
	train_accs = []
	test_accs = []
	epochs = []
	mdlroot = os.path.join(coderoot, "checkpoints", "VAE", "encoder")

	with open("VAErecord.csv", "w+") as f:
		lenthOfData = len(f.readlines())
		w = csv.writer(f)
		for file in os.listdir(mdlroot):
			if lenthOfData > 0:
				lenthOfData -= 1
				continue
			e = int("".join(filter(str.isdigit, file)))
			train_acc, test_acc = mae_test(neck_axis = 2,
			                               save_mdlfile = os.path.join(mdlroot,
			                                                           file))
			train_accs.append(train_acc)
			epochs.append(e)
			test_accs.append(test_acc)
			w.writerow([e, train_acc, test_acc])
	vis = visdom.Visdom()
	train_accs = torch.tensor(train_accs)
	test_accs = torch.tensor(test_accs)

	epochs = torch.tensor(epochs)
	vis.scatter(torch.stack((epochs, train_accs,), dim = 0).T, win = "train_acc")
	vis.scatter(torch.stack((epochs, test_accs,), dim = 0).T, win = "test_acc")
# mae_test(AutoEncoder = VAE_conv4)
