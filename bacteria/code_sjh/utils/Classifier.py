import csv

import numpy

from bacteria.code_sjh.utils.RamanData import RamanDatasetCore, Raman_dirwise
from bacteria.code_sjh.models import BasicModule
from bacteria.code_sjh.utils.Validation.validation import grad_cam
import os
import torch
import shutil


def copy_filewise_classify(db: RamanDatasetCore,
                           model: BasicModule,
                           dst: str,
                           device: torch.device = None):
	model.eval()
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	root = db.root
	if not os.path.isdir(dst):
		os.makedirs(dst)
	model = model.to(device)
	label2name = db.label2name()
	for i in range(len(db)):
		raman, label = db[i]
		raman = torch.unsqueeze(raman, dim = 0).to(device)

		file = db.RamanFiles[i]
		src = os.path.join(root, file)
		filename = file.split(os.sep)[-1]
		with torch.no_grad():
			logits = model(raman)  # [1,n_c]

		pred = torch.squeeze(logits.argmax(dim = 1)).item()
		true_name = label2name[label.item()]
		pred_name = label2name[pred]
		dst_dir = os.path.join(dst, true_name + "2" + pred_name)
		if not os.path.isdir(dst_dir):
			os.makedirs(dst_dir)
		dst_p = os.path.join(dst_dir, filename)
		shutil.copy(src, dst_p)


def cam_output_filewise(db: RamanDatasetCore,
                        model: BasicModule,
                        dst: str,
                        device: torch.device = None):
	model.train()
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# root = db.root
	if not os.path.isdir(dst):
		os.makedirs(dst)
	model = model.to(device)
	# label2name = db.label2name()
	if not os.path.isdir(dst):
		os.makedirs(dst)
	for i in range(len(db)):
		raman, label = db[i]
		raman = torch.unsqueeze(raman, dim = 0).to(device)
		file = db.RamanFiles[i]
		filename = file.split(os.sep)[-1][:-4] + ".cam.csv"
		cam = grad_cam(model, raman, label = None)
		dst_p = os.path.join(dst, filename)
		numpy.savetxt(dst_p, cam, delimiter = ",")


def report_dirwise_classify(dataset: Raman_dirwise,  # raman_dir -> root/class/dir/filename
                            model: BasicModule,
                            dst: str,
                            device: torch.device = None,
                            label2name = None,
                            stat_strategy = "file_wise",
                            ):
	"""
	生成每个文件夹对应的分类结果报告：
		流程：
			加载网络
			对db的每个数据进行分类（root/class/sample/filename）
			统计每个sample文件夹下各个文件（root/class/sample）的分类结果

	@param dataset:
	@param model:
	@param dst:
	@param device:
	@param label2name:
	@param stat_strategy:
		"file_wise": use argmax to predict
		"pred_wise": use logits to report
	@return:
	"""

	model.eval()
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	root = dataset.root

	model = model.to(device)
	if label2name is None:
		label2name = dataset.label2name()
	sample2label = dataset.sample2label()
	sample2data = dataset.get_data_sorted_by_sample()
	f = open(dst, "w", newline = "")
	writer = csv.writer(f)
	header = ["sample"]
	for label in label2name.keys():
		header.append(label2name[label])
	header.append("sum")
	header.append("true_label")
	writer.writerow(header)
	for sample in sample2data.keys():
		res = [sample] + [0] * (len(header) - 2)
		raman = sample2data[sample].to(device)
		with torch.no_grad():
			logits = model(raman)  # [num_file,n_c]
		if stat_strategy == "file_wise":
			pred = torch.detach(logits.argmax(dim = 1)).cpu().np()  # [num_file]
			for i in range(len(pred)):
				res[pred[i] + 1] += 1
				res[-1] += 1
		else:
			pred = torch.detach(logits.sum(dim = 0)).numpy()  # [n_c]
			for i in range(len(pred)):
				res[i + 1] = pred[i]
			res[-1] = pred.sum()

		assert sum(res[1:-1]) == res[-1]
		truelabelname = sample2label[sample].item()
		truelabelname = dataset.label2name()[truelabelname]
		res.append(truelabelname)
		writer.writerow(res)


if __name__ == '__main__':
	from bacteria.code_sjh.Core.basic_functions.fileReader import getRamanFromFile
	from bacteria.code_sjh.utils.Process_utils import Process
	from bacteria.code_sjh.models.CNN.AlexNet import AlexNet_Sun

	readdatafunc = getRamanFromFile(  # 定义读取数据的函数
		wavelengthstart = 39, wavelengthend = 1810, delimeter = None,
		dataname2idx = {"Wavelength": 0, "Intensity": 1}
	)

	dataroot = r"D:\myPrograms\pythonProject\Raman_dl_ml\bacteria\data\脑胶质瘤\data_indep"
	mdl_root = r"D:\myPrograms\pythonProject\Raman_dl_ml\bacteria\code_sjh\checkpoints\alexnet.mdl"

	db_cfg = dict(  # 数据集设置
		dataroot = dataroot,
		backEnd = ".csv",
		# backEnd = ".asc",
		t_v_t = [0.8, 0.1, 0.1],
		LoadCsvFile = readdatafunc,
		k_split = 9,
		transform = Process.process_series([  # 设置预处理流程
			Process.interpolator(),
			Process.sg_filter(),
			Process.bg_removal_niter_fit(),
			Process.norm_func(), ]
		))
	dataset = Raman_dirwise(mode = "all", **db_cfg)
	sample_tensor = torch.unsqueeze(dataset[0][0], dim = 0)
	net = AlexNet_Sun(sample_tensor, num_classes = 2)
	net.load(mdl_root)
	sample2name = dataset.get_data_sorted_by_sample()
	report_dirwise_classify(dataset, net, "res.csv", label2name = {0: "neg", 1: "pos"})
