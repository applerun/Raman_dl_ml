import os, sys

coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "data_AST")
sys.path.append(coderoot)
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


from bacteria.code_sjh.models import BasicModule



def train(model: BasicModule,
          lr: float,
          device: torch.device,
          train_loader: DataLoader,
          criteon,
          optimizer,
          idx,
          args = None,
          snapshot_path = None,
          verbose = False,
          mixed = False,

          loss_plus: callable = None,
          lr_decay_rate = 0,
          lr_decay_period = 30):
	"""

	:param model: 需要训练的模型
	:param lr: 基础学习率
	:param device: 在gpu或cpu上训练模型
	:param train_loader: 训练数据集
	:param criteon: 损失函数
	:param optimizer: 优化器
	:param idx: 此次训练是第idx个epoch
	:param args: TODO:其它设定
	:param snapshot_path: 中继点存储路径
	:param verbose: 是否打印该epoch中的详细信息
	:param mixed: 混合精度训练
	:param unsupervised: 是否为无监督学习（True：label=
	:param loss_plus: 如果网络forward输出多于一个值，请加入loss_plus以计算新的结果
	:param lr_decay_rate:
	:param lr_decay_period:
	:return:
	"""

	model.train()
	# lr decay
	# not_decay_flag = idx // lr_decay_period
	if 0 < lr_decay_rate < 1 and idx % lr_decay_period == 0:
		lr = lr * lr_decay_rate ** (idx // lr_decay_period)
		for param_g in optimizer.param_groups:
			param_g["lr"] = lr
	print('\nEpoch {} starts, please wait...'.format(idx))
	loader = tqdm(train_loader)
	loss_list = []
	scaler = GradScaler()  # 混合精度训练
	# t = "scheduler" if type(optimizer) == torch.optim.lr_scheduler.ReduceLROnPlateau else "optimizer"
	for step, (spectrum, label) in enumerate(loader):
		spectrum, label = spectrum.to(device, non_blocking = True), label.to(device, non_blocking = True)
		if mixed:
			with autocast():
				output = model(spectrum)
				loss = criteon(output, label) if loss_plus is None else criteon(output[0], label) + loss_plus(
					*output[1:])
				with torch.no_grad():
					print("cri:", criteon(output[0], label), "\nkld:", loss_plus(
						*output[1:]))
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad()
		else:
			output = model(spectrum)
			loss = criteon(output, label) if loss_plus is None else criteon(output[0], label) + loss_plus(*output[1:])
			# with torch.no_grad():
			# 	if loss_plus is not None:
			# 		print("cri:", criteon(output[0], label), "\nkld:", loss_plus(
			# 			*output[1:]))
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		loss_list.append(loss.item())

		loader.set_postfix_str(
			'lr:{:.8f}, loss: {:.4f}'.format(optimizer.param_groups[0]['lr'], np.mean(loss_list)))

	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	if verbose:
		print('[ Training ] Lr:{:.8f}, Epoch Loss: {:.4f}'.format(optimizer.param_groups[0]['lr'], np.mean(loss_list)))
	return np.mean(loss_list)


def train_CVAE(model: BasicModule,
               train_loader: DataLoader,
               criteon,
               optimizer,
               idx,
               criteon_label = None,
               verbose = False,
               mixed = False,
               device: torch.device = None,
               lr: float = 0.0001,
               lr_decay_rate = 0,
               lr_decay_period = 30,
               label_rate = 50.,
               kld_rate = 200.0,
               ):
	"""

	:param model: 需要训练的模型
	:param lr: 基础学习率
	:param device: 在gpu或cpu上训练模型
	:param train_loader: 训练数据集
	:param criteon: 损失函数
	:param optimizer: 优化器
	:param idx: 此次训练是第idx个epoch
	:param args: TODO:其它设定
	:param snapshot_path: 中继点存储路径
	:param verbose: 是否打印该epoch中的详细信息
	:param mixed: 混合精度训练
	:param unsupervised: 是否为无监督学习（True：label=
	:param loss_plus: 如果网络forward输出多于一个值，请加入loss_plus以计算新的结果
	:param lr_decay_rate:
	:param lr_decay_period:
	:return:
	"""
	if device == None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.train()
	if criteon_label is None:
		criteon_label = nn.CrossEntropyLoss()
	# lr decay
	not_decay_flag = idx // lr_decay_period
	if 0 < lr_decay_rate < 1 and idx % lr_decay_period == 0:
		lr = lr * lr_decay_rate ** (idx // lr_decay_period)
		for param_g in optimizer.param_groups:
			param_g["lr"] = lr

	# process bar
	if verbose:
		print('\nEpoch {} starts, please wait...'.format(idx))
		loader = tqdm(train_loader)
	else:
		loader = train_loader
	# loss
	loss_list = []

	# train
	# 混合精度训练
	for step, (spectrum, label) in enumerate(loader):
		spectrum = spectrum.to(device, non_blocking = True)
		if mixed:
			scaler = GradScaler()
			with autocast():
				x_hat, y_c_hat, kld = model(spectrum, label)
				loss1 = criteon(
					x_hat,
					spectrum
				)
				loss2 = criteon_label(y_c_hat,label)
				loss = loss1 + label_rate * loss2 + kld_rate * kld
				if verbose:
					with torch.no_grad():
						print("cri:", loss, "\nkld:", kld)
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad()
		else:
			x_hat, y_c_hat, kld = model(spectrum, label)
			loss1 = criteon(
				x_hat,
				spectrum
			)
			loss2 = criteon_label(y_c_hat, label.to(device))
			# print("loss_hat:",loss1.cpu().detach().numpy())
			# print("loss_label:",loss2.cpu().detach().numpy())
			loss = loss1 + label_rate * loss2 + kld_rate * kld
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		loss_list.append(loss.item())
		if verbose:
			loader.set_postfix_str(
				'lr:{:.8f}, loss: {:.4f}'.format(optimizer.param_groups[0]['lr'], np.mean(loss_list)))

	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	if verbose:
		print('[ Training ] Lr:{:.8f}, Epoch Loss: {:.4f}'.format(optimizer.param_groups[0]['lr'], np.mean(loss_list)))
	return np.mean(loss_list)
