import csv
import glob
import os
import random
import sys

import numpy
import visdom

import torch
import torch.utils.data
from torch.utils.data import Dataset

coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "data_ID")
sys.path.append(coderoot)
try:
	from ..utils import Process
	from ..utils.Validation import visdom_utils
except:
	from utils import Process
	from utils.Validation import visdom_utils


# 请将combined文件后缀前加上减号" - "

def LoadCombinedFile(filename: str, encoding = "utf-8-sig"):
	"""
	本函数用于读取数据
	:param encoding:
	:param filename: data file name
	:return:
	"""
	if not filename.endswith(".csv"):
		filename += ".csv"

	with open(filename, "r", encoding = encoding) as f:
		lines = f.readlines()

		for line in lines:
			data = [float(i) for i in line.split(",")]  # 将字符串转化为浮点数
			yield data
	return None


def getLamenFromFile(wavelengthstart = 400, wavelengthend = 1600, dataname2idx = None):
	if dataname2idx is None:
		dataname2idx = {"Wavelength": 0, "Column": 1, "Intensity": 2}
	if wavelengthend < wavelengthstart:
		wavelengthstart, wavelengthend = wavelengthend, wavelengthstart

	def func(filepath):
		Ramans = []
		Wavelengths = []
		with open(filepath, "r") as f:
			lines = f.readlines()
			for line in lines:
				data = line.split(",")
				if data[0] in ["ROI", "Wavelength", "Column", "Intensity"]:
					continue
				wavelength = float(data[dataname2idx["Wavelength"]])
				intense = float(data[dataname2idx["Intensity"]])
				if wavelengthstart < wavelength and wavelength < wavelengthend:
					Ramans.append(intense)
					Wavelengths.append(wavelength)
				elif wavelength > wavelengthend:
					break
		return Ramans, Wavelengths

	return func


class test(Dataset):
	def __init__(self):
		super(test, self).__init__()
		self.data = [list(range(1000)) for i in range(720)]
		self.label = [i % 6 for i in range(720)]

	def __len__(self):
		return 720

	def __getitem__(self, item):
		return self.data[item], self.label[item]


class Raman(Dataset):
	def __init__(self,
	             dataroot: str = dataroot,
	             resize = None,
	             mode = "train",
	             t_v_t = None,
	             savefilemode: str = "combinedfile",
	             sfpath = "Ramans.csv",
	             shuffle = True,
	             transform = Process.preprocess_default,
	             LoadCsvFile = None,
	             backEnd = "-p.csv",
	             unsupervised: bool = False,
	             noising = None,
	             newfile = False,

	             ):
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
		super(Raman, self).__init__()

		if t_v_t is None:  # 分割train-validation-test
			t_v_t = [0.7, 0.2, 0.1]
		# assert t_v_t[0] + t_v_t[1] <= 1
		self.new = newfile
		self.train_split = t_v_t[0]
		self.validation_split = t_v_t[1]
		self.test_split = t_v_t[2]
		self.LoadCsvFile = LoadCsvFile
		self.root = dataroot
		self.resize = resize
		self.sfmode = savefilemode
		self.name2label = {}  # 为每个分类创建一个label
		self.sfpath = sfpath
		self.shuff = shuffle
		self.mode = mode
		self.dataEnd = backEnd
		self.transform = transform
		self.unsupervised = unsupervised
		self.noising = noising
		self.xs = None
		if self.LoadCsvFile is None:
			if self.sfmode == "combinedfile":
				self.LoadCsvFile = LoadCombinedFile
			elif self.sfmode == "1lamen1file":
				self.LoadCsvFile = getLamenFromFile()

		for name in sorted(os.listdir(dataroot)):
			if not os.path.isdir(os.path.join(dataroot, name)):
				continue
			if not len(os.listdir(os.path.join(dataroot, name))):
				continue
			self.name2label[name] = len(self.name2label.keys())

		self.numclasses = len(self.name2label.keys())
		self.Ramans, self.labels = self.LoadCsv(sfpath)  # 所有的数据和文件

		# 数据分割
		if mode == "train":  # 分割训练集
			self.Ramans = self.Ramans[:int(self.train_split * len(self.Ramans))]
			self.labels = self.labels[:int(self.train_split * len(self.labels))]
		elif mode == "val":  # 分割验证集
			self.Ramans = self.Ramans[int(self.train_split * len(self.Ramans)):
			                          int((self.train_split + self.validation_split) * len(self.Ramans))]
			self.labels = self.labels[int(self.train_split * len(self.labels)):
			                          int((self.train_split + self.validation_split) * len(self.labels))]
		elif mode == "test":  # 分割测试集
			self.Ramans = self.Ramans[int((self.train_split + self.validation_split) * len(self.Ramans)):]
			self.labels = self.labels[int((self.train_split + self.validation_split) * len(self.labels)):]
		else:
			raise Exception("Invalid mode!", mode)

	def shuffle(self):
		z = list(zip(self.Ramans, self.labels))
		random.shuffle(z)
		self.Ramans[:], self.labels[:] = zip(*z)
		return

	def shufflecsv(self, filename = None):
		if filename is None:
			filename = self.sfpath
		path = os.path.join(self.root, filename)
		with open(path, "r") as f:
			reader = csv.reader(f)
			rows = []
			for row in reader:
				rows.append(row)
			random.shuffle(rows)
		with open(path, "w", newline = "") as f:
			writer = csv.writer(f)

			writer.writerows(rows)

		return

	def num_classes(self):
		return self.numclasses

	def name2label(self):
		return self.name2label

	def label2name(self):
		keys = list(self.name2label.values())
		values = list(self.name2label.keys())
		return dict(zip(keys, values))

	def LoadCsv(self, filename, ):
		"""
		用于对数据分类并记录
		:param filename: 记录文件路径
		:return: 文件记录的所有lamen spectrum 和 label
		"""

		if not os.path.exists(os.path.join(self.root, filename)) or self.new:
			if self.sfmode == "1lamen1file":  # 每个光谱一个文件
				Ramans = []
				for name in self.name2label.keys():
					Ramans += glob.glob(os.path.join(self.root, name, "*.csv"))

				if self.shuff:  # 打乱顺序
					random.shuffle(Ramans)
				self.RamanNames = Ramans
				with open(os.path.join(self.root, filename), mode = "w", newline = "") as f:  # 记录所有数据
					writer = csv.writer(f)
					for spectrum in Ramans:  # spectrum:data root/label name/**.csv
						name = spectrum.split(os.sep)[-2]  # label name
						label = self.name2label[name]  # label idx
						R,X = self.LoadCsvFile(spectrum)
						writer.writerow([label] + R)
						if self.xs == None:
							self.xs = X
				with open(os.path.join(self.root, filename.split(".")[0] + "_xs." + filename.split(".")[1]), mode = "w",
				          newline = "") as f:
					writer = csv.writer(f)
					writer.writerow(self.xs)

			if self.sfmode == "combinedfile":
				Ramans = []
				labels = []
				for name in self.name2label.keys():

					conbined_file_l = glob.glob(os.path.join(self.root, name, "*" + self.dataEnd))  # 找到所有的conbinedfile
					for files in conbined_file_l:
						datas = self.LoadCsvFile(files)
						if self.xs == None:
							self.xs = datas.__next__()
						for spectrum in datas:
							if not spectrum:
								continue
							Ramans.append(spectrum)
							labels.append(self.name2label[name])
				if self.shuff:  # 打乱
					s = list(zip(Ramans, labels))
					random.shuffle(s)
					Ramans[:], labels[:] = zip(*s)

				with open(os.path.join(self.root, filename), "w", newline = "") as f:
					csvwriter = csv.writer(f)
					for i in range(len(labels)):
						csvwriter.writerow([labels[i]] + Ramans[i])
				with open(os.path.join(self.root, filename.split(".")[0] + "_xs." + filename.split(".")[1]),
				          mode = "w", newline = "") as f:
					writer = csv.writer(f)
					writer.writerow(self.xs)
		Ramans = []
		labels = []

		with open(os.path.join(self.root, filename)) as f:
			reader = csv.reader(f)
			for row in reader:
				label = int(row[0])
				spectrum = [float(i) for i in row[1:]]
				spectrum = numpy.array(spectrum)

				# 数据预处理
				if self.transform is not None:
					spectrum = self.transform(spectrum)

				labels.append(torch.tensor(label))
				Ramans.append(torch.unsqueeze(torch.tensor(spectrum).to(torch.float32), dim = 0).to(torch.float32))
		if self.xs == None:

			with open(os.path.join(self.root, filename.split(".")[0] + "_xs." + filename.split(".")[1])) as f:
				xs = f.readline().split(",")
				self.xs = [float(x) for x in xs]
				assert len(self.xs) == len(Ramans[0])
		assert len(Ramans) == len(labels)
		return Ramans, labels  # [[float],[],...[]],[int]

	def get_data_sorted_by_label(self):
		"""
		返回一个字典，每个label对应的数字指向一个tensor[label=label的光谱的数量,c=1,lenth]
		"""
		spectrum_each_label = {}
		for i in range(len(self.name2label)):
			spectrum_each_label[i] = None
		for i in range(self.__len__()):
			lamen, label = self.Ramans[i], self.labels[i]
			if spectrum_each_label[label.item()] is None:
				spectrum_each_label[label.item()] = torch.unsqueeze(lamen, dim = 0)
			else:
				spectrum_each_label[label.item()] = torch.cat(
					(spectrum_each_label[label.item()],
					 torch.unsqueeze(lamen,
					                 dim = 0)),
					dim = 0
				)

		return spectrum_each_label

	def wavelengths(self):
		return torch.tensor(self.xs)

	def show_data(self, xs = None, vis = None, win = "data"):
		if vis is None:
			vis = visdom.Visdom()
		label2name = self.label2name()
		label2data = self.get_data_sorted_by_label()
		if xs == None:
			xs = self.wavelengths()
		if label2data[0].shape[-1] > 3:  # 光谱数据
			for i in range(self.numclasses):
				data = label2data[i]
				name = label2name[i]
				visdom_utils.spectrum_vis(data, win = win + "_" + name, xs = xs, vis = vis)
		else:  # 降维后数据
			for i in range(self.numclasses):
				data = torch.squeeze(label2data[i])
				vis.scatter(data,
				            win = win,
				            update = None if i == 0 else "append",
				            name = self.label2name()[i],
				            opts = dict(
					            title = win,
					            showlegend = True,
				            )
				            )
		return

	def __len__(self):
		return len(self.Ramans)

	def __getitem__(self, item):

		lamen, label = self.Ramans[item], self.labels[item]
		lamen = lamen.to(torch.float32)
		if self.unsupervised:  # label设置为noising前的光谱数据
			label = lamen.to(torch.float32)
			if not self.noising == None and self.mode == "train":
				lamen = torch.squeeze(lamen)
				lamen = self.noising(lamen)
				lamen = torch.unsqueeze(lamen, dim = 0).to(torch.float32)

		return lamen, label


def pytorchlize(x):
	for i in range(3 - len(x.shape)):
		x = torch.unsqueeze(x, dim = len(x.shape) - 1)
	return x


if __name__ == '__main__':
	visdom_utils.startVisdomServer()
	Ramanroot = os.path.join(os.path.split(os.path.split(coderoot)[0])[0], "data", "liver", "20220301")
	db = Raman(Ramanroot, savefilemode = "1lamen1file",
	           LoadCsvFile = getLamenFromFile(wavelengthstart = 400, wavelengthend = 1800,
	                                          dataname2idx = {"Wavelength": 0, "Column": 1, "Intensity": 2}),
	           newfile = True, t_v_t = [1, 0, 0])
	db.show_data()
