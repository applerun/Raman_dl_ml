import os, numpy as np, csv, time

import torch, visdom
from torch import nn, optim
from torch.utils.data import DataLoader
from bacteria.code_sjh.utils.iterator import train_CVAE
from bacteria.code_sjh.utils import Process
from bacteria.code_sjh.utils.Validation import visdom_utils as validation, visdom_utils
from bacteria.code_sjh.models import CVAE, CVAE2_Dlabel_Dclassifier, RamanData, projectroot, coderoot

# eval
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, confusion_matrix
from conv_net_classify import plt_res_val, plt_loss_acc, heatmap
import matplotlib.pyplot as plt


def plt_h(pltdir,
          res,
          label2name,
          informations = None):
	if informations is None:
		informations = ""
	label2h = res["label2h"]
	fig, ax = plt.subplots()
	for label in label2h.keys():
		h = label2h[label]
		if type(h) == torch.Tensor:
			h = h.detach_().cpu().numpy()
		name = label2name[label]
		ax.scatter(h[:,0], h[:,1], label = name)
		if not os.path.isdir(os.path.join(pltdir, informations + "_record")):
			os.makedirs(os.path.join(pltdir, informations + "_record"))
		np.savetxt(os.path.join(pltdir, informations + "_record", name + "_VAE_neck.csv"), h, delimiter = ",")
	ax.set_title('VAE latent space')
	ax.set_xlabel('VAE1')
	ax.set_ylabel('VAE2')
	ax.legend()
	fig.savefig(os.path.join(pltdir, informations + "_neck_vis.png"))
	plt.close(fig)
	return


def plt_res(pltdir,
            res,
            val_db,
            informations = None):
	if informations is None:
		informations = ""
	label2name = val_db.label2name()
	plt_loss_acc(pltdir, res, informations)
	plt_res_val(pltdir, res["res_val"], label2name, informations = "val")
	plt_res_val(pltdir, res["res_test"], label2name, informations = "test")
	plt_h(pltdir, res["res_val"], label2name, informations = "val")
	plt_h(pltdir, res["res_test"], label2name, informations = "test")


def eval_CVAE(model,
              loader,
              device):
	model.eval()
	correct = 0
	total = len(loader.dataset)
	for x, y in loader:
		x = x.to(device)
		with torch.no_grad():
			_, logits, __ = model(x, y)
			pred = logits.argmax(dim = 1)
		c_t = torch.eq(pred, y.to(device)).sum().float().item()
		correct += c_t
	acc = correct / total
	return acc


def evaluate_CVAE_all(model: CVAE,
                      loader,
                      criteon,
                      device: torch.device,
                      criteon_label = None,
                      label_rate = 50.,
                      kld_rate = 200.0,
                      ):
	if criteon_label is None:
		criteon_label = nn.CrossEntropyLoss()
	model.eval()
	correct = 0
	total = len(loader.dataset)
	num_clases = loader.dataset.num_classes()
	loss_list = []
	y_true_all = {}
	y_score_all = {}
	conf_m = np.zeros((num_clases, num_clases))
	for i in range(num_clases):
		y_score_all[i] = []
		y_true_all[i] = []

	with torch.no_grad():
		for x, y in loader:
			x, y = x.to(device), y.to(device)
			with torch.no_grad():
				x_hat, y_c_hat, kld = model(x, y)
				pred = y_c_hat.argmax(dim = 1)
				c_t = torch.eq(pred, y.to(device)).sum().float().item()
				loss1 = criteon(
					x_hat,
					x
				)
				loss2 = criteon_label(y_c_hat, y)
				loss = loss1 + label_rate * loss2 + kld_rate * kld
				loss_list.append(loss.item())
				correct += c_t
				score = F.softmax(y_c_hat, dim = 1)
				for i in range(num_clases):
					y_true = torch.eq(y, i).int().cpu().numpy()
					y_score = score[:, i].float().cpu().numpy()
					y_true_all[i] = np.append(y_true_all[i], y_true)
					y_score_all[i] = np.append(y_score_all[i], y_score)
				cm = confusion_matrix(y.cpu().numpy(), pred.cpu().numpy(), labels = list(range(num_clases)))
				conf_m += cm
		label2data = loader.dataset.get_data_sorted_by_label()

		label2roc = {}
		label2auc = {}
		label2h = {}

		for i in range(num_clases):
			x = label2data[i]
			input = x.to(device)
			h = model.encode(input, rand = 0.5)  # scatter
			frp, tpr, thresholds = roc_curve(y_true_all[i], y_score_all[i])
			label2roc[i] = (frp, tpr, thresholds)
			label2auc[i] = auc(frp, tpr)
			label2h[i] = h.cpu().numpy()
		acc = correct / total
		loss = np.mean(loss_list)
		res = dict(acc = acc, loss = loss, label2roc = label2roc, label2auc = label2auc, label2h = label2h,
		           confusion_matrix = conf_m)

	return res


def evaluate_CVAE_loss(model,
                       val_loader,
                       criteon,
                       device,
                       criteon_label = None,
                       label_rate = 50.,
                       kld_rate = 200.0, ):
	if criteon_label is None:
		criteon_label = nn.CrossEntropyLoss()
	model.eval()
	loss_list = []
	with torch.no_grad():
		for spectrum, label in val_loader:
			spectrum, label = spectrum.to(device), label.to(device)
			x_hat, y_c_hat, kld = model(spectrum, label)
			loss1 = criteon(
				x_hat,
				spectrum
			)
			loss2 = criteon_label(y_c_hat, label)
			loss = loss1 + label_rate * loss2 + kld_rate * kld
			loss_list.append(loss.item())
	return np.mean(loss_list)


def main(model,
         train_db,
         val_db,
         test_db,
         criteon,
         batchsz = 40,
         numworkers = 0,
         lr = 0.0001,
         viz = None,
         save_dir = os.path.join(coderoot, "checkpoints", ),
         epochs = 200,
         device = None,
         epo_interv = 30,
         rates = None,
         verbose = False,
         ):
	if rates is None:
		rates = dict(
			label_rate = 50.,
			kld_rate = 200.0,
		)
	if viz == None:
		viz = visdom.Visdom()

	if device == None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_losses, val_losses, test_losses = [], [], []
	train_acces, val_acces, test_accses = [], [], []
	steps = []
	best_acc, best_epoch = 0, 0
	global_step = 0
	viz.line([0], [-1], win = "loss_" + str(k), opts = dict(title = "loss_" + str(k)))
	viz.line([0], [-1], win = "train_acc_" + str(k), opts = dict(title = "train_acc_" + str(k)))
	viz.line([0], [-1], win = "val_acc_" + str(k), opts = dict(title = "val_acc_" + str(k)))
	if not os.path.isdir(save_dir):
		os.mkdir(save_dir)
	model.model_loaded = True
	# save_path = os.path.join(save_dir, model.model_name + ".mdl")
	# if not os.path.exists(save_path):
	# 	with open(save_path, "w", newline = ""):
	# 		pass

	optimizer = optim.Adam(model.parameters(), lr = lr)
	train_loader = DataLoader(train_db, batch_size = batchsz, shuffle = True, num_workers = numworkers,
	                          )
	val_loader = DataLoader(val_db, batch_size = batchsz, num_workers = numworkers)
	test_loader = DataLoader(test_db, batch_size = batchsz, num_workers = numworkers)
	label2data_val = val_db.get_data_sorted_by_label()
	sample2data_val = val_db.get_data_sorted_by_sample()
	label2data_train = train_db.get_data_sorted_by_label()
	# label2data_train = train_db.get_data_sorted_by_sample()
	label2name = val_db.label2name()
	xs = torch.tensor(val_db.xs)
	for epoch in range(epochs):
		loss = train_CVAE(model, lr = lr, device = device, train_loader = train_loader, criteon = criteon,
		                  optimizer = optimizer, idx = epoch, **rates)
		viz.line([loss.item()], [global_step], win = "loss_" + str(k), update = "append")
		global_step += 1
		if epoch % epo_interv == epo_interv - 1 or epoch == epochs - 1:
			model.eval()

			val_acc = eval_CVAE(model, val_loader, device)
			train_acc = eval_CVAE(model, train_loader, device)
			val_loss = evaluate_CVAE_loss(model, val_loader, criteon, device, **rates)
			# print("epoch:{},acc:{}".format(epoch, val_acc))
			if val_acc >= best_acc and epoch > 100:  # Early stopping
				model.save(os.path.join(save_dir, "trained", "epoch{}.mdl".format(epoch + 1)))
				model.Encoder.save(os.path.join(save_dir, "encoder", "epoch{}.mdl".format(epoch + 1)))
				best_acc = val_acc
				best_epoch = epoch + 1

			viz.line([val_acc], [global_step], win = "val_acc_" + str(k), update = "append")
			viz.line([train_acc], [global_step], win = "train_acc_" + str(k), update = "append")

			train_acces.append(train_acc)
			val_acces.append(val_acc)
			# test_accses.append(test_acc)
			train_losses.append(loss)
			val_losses.append(val_loss)
			# test_losses.append(test_loss)
			steps.append(global_step)
		if verbose and (epoch % 1000 == 999 or epoch == epochs - 1):
			with torch.no_grad():
				newwin = 0
				for samplename in sample2data_val.keys():
					spectrum = sample2data_val[samplename].to(device)
					if model.neck_axis < 4:
						model.neck_vis(spectrum, samplename,
						               win = "VAE_val_neck_vis_samplewise_epoch:" + str(epoch + 1),
						               vis = viz,
						               update = None if newwin == 0 else "append", rand = 0.5)
						newwin = 1
				for idx in range(len(label2name.keys())):

					spectrum = label2data_val[idx]
					spectrum = spectrum.to(device)
					spectrum_hat = model(spectrum)[0]
					win_main = "spectrum_VAE_val_" + label2name[idx] + "_epoch:" + str(epoch + 1)
					win = win_main + label2name[idx]

					validation.spectrum_vis(spectrum, xs, win, name = "x", vis = viz)
					validation.spectrum_vis(spectrum_hat, xs, win, update = "append", name = "x_hat", vis = viz)
					if model.neck_axis < 4:
						model.neck_vis(spectrum, label2name[idx], win = "VAE_val_neck_vis_epoch:" + str(epoch + 1),
						               vis = viz,
						               update = None if idx == 0 else "append")
					# with torch.no_grad():
					spectrum = label2data_train[idx]
					spectrum = spectrum.to(device)
					spectrum_hat = model(spectrum)[0]
					win = "spectrum_VAE_train_" + label2name[idx] + "_epoch:" + str(epoch + 1)
					validation.spectrum_vis(spectrum, xs, win, name = "x", vis = viz)
					validation.spectrum_vis(spectrum_hat, xs, win, update = "append", name = "x_hat", vis = viz)
					if model.neck_axis < 4:
						model.neck_vis(spectrum, label2name[idx], vis = viz,
						               update = None if idx == 0 else "append",
						               win = "VAE_train_neck_vis_epoch:" + str(epoch + 1))
	model.load(os.path.join(save_dir, "trained", "epoch{}.mdl".format(best_epoch)))
	test_acc = eval_CVAE(model, test_loader, device)
	res_val = evaluate_CVAE_all(model, val_loader, criteon, device, **rates)
	res_test = evaluate_CVAE_all(model, test_loader, criteon, device, **rates)
	res = dict(train_acces = train_acces, val_acces = val_acces,  # 训练过程——正确率
	           train_losses = train_losses, val_losses = val_losses,  # 训练过程——损失函数
	           best_acc = best_acc, test_acc = test_acc, best_epoch = best_epoch,
	           res_val = res_val, res_test = res_test
	           )
	return res


if __name__ == '__main__':
	from scipy import interpolate

	visdom_utils.startVisdomServer()
	vis = visdom.Visdom()  # visdom对象
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运算设备
	raman = RamanData.Raman_dirwise

	readdatafunc0 = RamanData.getRamanFromFile(wavelengthstart = 596, wavelengthend = 1802,
	                                           dataname2idx = {"Wavelength": 0, "Column": 1, "Intensity": 2})


	def readdatafunc(filepath):
		R, X = readdatafunc0(filepath)
		f = interpolate.interp1d(X, R, kind = "cubic")
		newX = np.linspace(600, 1800, 517)
		newR = f(newX)
		return newR, newX



	k_split = 6
	db_cfg = dict(
		# dataroot = os.path.join(projectroot, "data", "data_AST")
		# dataroot = os.path.join(projectroot, "data", "liver", "liver_all_samplewise")
		dataroot = os.path.join(projectroot, "data", "liver_cell_dou"), backEnd = ".csv", t_v_t = [0.8, 0.2, 0.0],
		LoadCsvFile = readdatafunc,k_split = k_split)

	train_cfg = dict(
		criteon = nn.BCELoss(),
		epochs = 5000,
		batchsz = 256,
		lr = 0.0001, rates = dict(
			label_rate = 50.,
			kld_rate = 200.0,
		),
		viz = vis,
		device = device
	)
	n_iter = 1  # 交叉验证重复次数
	recorddir = "Record_" + time.strftime("%Y-%m-%d-%H_%M_%S")
	recorddir = os.path.join(projectroot, "results", "liver_CVAE", recorddir)  # 实验结果保存位置
	if not os.path.isdir(recorddir):
		os.makedirs(recorddir)
	label_rates = [10,20,50,100]
	kld_rates = [20,50,100,200]
	rate2acc = np.zeros((len(label_rates),len(kld_rates)))
	for i_l in range(len(label_rates)):
		label_rate = label_rates[i_l]
		for i_k in range(len(kld_rates)):
			kld_rate = kld_rates[i_k]
			bestaccs, testaccs, bepochs, vaucs, taucs = [], [], [], [], []
			train_cfg["rates"] = dict(label_rate= label_rate,kld_rate = kld_rate)
			for Model in [CVAE2_Dlabel_Dclassifier]:
				recordsubdir = os.path.join(recorddir,
				                            "CVAERecord" + time.asctime().replace(":", "-").replace(" ", "_"))  # 每个模型一个文件夹保存结果
				if not os.path.isdir(recordsubdir):
					os.makedirs(recordsubdir)
				recordfile = recordsubdir + ".csv"
				f = open(recordfile, "w", newline = "")
				writer = csv.writer(f)
				f.write(db_cfg.__str__() + "\n")
				f.write(train_cfg.__str__() + "\n")
				writer.writerow(["n", "k", "bestaccs", "testaccs", "best_epoch", "val_AUC", "test_AUC"])
				conf_m_v = None
				conf_m_t = None
				for n in range(n_iter):
					sfpath = "Raman_" + str(n) + ".csv"
					for k in range(k_split):

						train_db = raman(**db_cfg, mode = "train", k = k, sfpath = sfpath)
						val_db = raman(**db_cfg, mode = "val", k = k, sfpath = sfpath)
						# test_db = raman(**db_cfg, mode = "test", k = k)
						if conf_m_t is None:
							conf_m_t = np.zeros((train_db.numclasses, train_db.numclasses))
							conf_m_v = np.zeros((train_db.numclasses, train_db.numclasses))
						s_t, s_l = train_db[0]
						s_t = torch.unsqueeze(s_t, dim = 0)
						n_c = train_db.numclasses

						model = Model(s_t, n_c).to(device)

						res = main(model, train_db = train_db, val_db = val_db, test_db = val_db, **train_cfg)

						pltdir = os.path.join(recordsubdir, "n-{}-k-{}".format(n, k))
						if not os.path.isdir(pltdir):
							os.makedirs(pltdir)
						b, t, be, auc_val, auc_test = res["best_acc"], res["res_test"]["acc"], res["best_epoch"], \
						                              np.mean(list(res["res_val"]["label2auc"].values())), \
						                              np.mean(list(res["res_test"]["label2auc"].values()))
						bestaccs.append(b)
						testaccs.append(t)
						bepochs.append(be)
						vaucs.append(auc_val)
						taucs.append(auc_test)
						conf_m_v += res["res_val"]["confusion_matrix"]
						conf_m_t += res["res_test"]["confusion_matrix"]
						writer.writerow([n, k, b, t, be, auc_val, auc_test])
						plt_res(pltdir, res, val_db, informations = None)
			np.savetxt(os.path.join(recordsubdir, "test_confusion_matrix.csv"), conf_m_v, delimiter = ",")
			np.savetxt(os.path.join(recordsubdir, "val_confusion_matrix.csv"), conf_m_t, delimiter = ",")
			heatmap(conf_m_t, os.path.join(recordsubdir, "test_confusion_matrix.png"))
			heatmap(conf_m_v, os.path.join(recordsubdir, "val_confusion_matrix.png"))
			ba = np.mean(np.array(bestaccs)).__str__() + " +- " + np.std(np.array(bestaccs)).__str__()
			ta = np.mean(np.array(testaccs)).__str__() + " +- " + np.std(np.array(testaccs)).__str__()
			bea = np.mean(np.array(bepochs)).__str__() + "+-" + np.std(np.array(bepochs)).__str__()
			auc_av = np.mean(np.array(vaucs)).__str__() + "+-" + np.std(np.array(vaucs)).__str__()
			auc_at = np.mean(np.array(taucs)).__str__() + "+-" + np.std(np.array(taucs)).__str__()
			writer.writerow(["mean", "std", ba, ta, bea, auc_av, auc_at])
			f.close()

			print("best acc:", ba)
			print("test acc", ta)
