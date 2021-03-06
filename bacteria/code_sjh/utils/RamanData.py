import csv
import glob
import os
import random
import sys
import copy
import numpy
import visdom

import torch
import torch.utils.data
from torch.utils.data import Dataset

coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "data_ID")
sys.path.append(coderoot)
from sklearn.preprocessing import LabelBinarizer


from bacteria.code_sjh.utils import Process
from bacteria.code_sjh.utils.Validation import visdom_utils


# 请将combined文件后缀前加上减号" - "

def LoadCombinedFile(filename: str,
                     encoding = "utf-8-sig"):
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


def getRamanFromFile(wavelengthstart = 400,
                     wavelengthend = 1800,
                     dataname2idx = None,
                     delimeter = None):

	if wavelengthend < wavelengthstart:
		wavelengthstart, wavelengthend = wavelengthend, wavelengthstart

	def func(filepath:str,
	         delimeter = delimeter,
	         dataname2idx = dataname2idx):
		if dataname2idx is None:
			dataname2idx = {}
		Ramans = []
		Wavelengths = []
		if delimeter is None:
			if filepath.endswith(".csv"):
				delimeter = ","
			elif filepath.endswith(".asc"):
				delimeter = "\t"

		with open(filepath, "r") as f:
			lines = f.readlines()
			header = None
			for line in lines:
				line = line.strip()
				data = line.split(delimeter)

				if data[0] in ["ROI", "Wavelength", "Column", "Intensity"]:
					if header is None:
						header = data
						dataname2idx["Wavelength"] = header.index("Wavelength")
						dataname2idx["Intensity"] = header.index("Intensity")
					continue
				try:
					wavelength = float(data[dataname2idx["Wavelength"]])
					intense = float(data[dataname2idx["Intensity"]])
				except:
					print(filepath, ":", data,",delimeter:",delimeter)
					raise ValueError
				if wavelengthstart < wavelength and wavelength < wavelengthend:
					Ramans.append(intense)
					Wavelengths.append(wavelength)
				elif wavelength > wavelengthend:
					break
		Ramans = numpy.array([Ramans])
		Wavelengths = numpy.array(Wavelengths)
		return Ramans, Wavelengths

	return func


class RamanDatasetCore(Dataset):  # 增加了一些基础的DataSet功能
	def __init__(self,
	             dataroot: str,

	             mode = "train",
	             t_v_t = None,
	             sfpath = "Ramans.csv",
	             shuffle = True,
	             transform = Process.preprocess_default,
	             LoadCsvFile = None,
	             backEnd = ".csv",
	             unsupervised: bool = False,
	             noising = None,
	             newfile = False,
	             k_split: int = None,
	             k: int = 0,
	             ratio: dict = None,
	             ):
		"""

		:param dataroot: 数据的根目录
		:param resize: 光谱长度(未实装)
		:param mode: "train":训练集 "val":验证集 "test":测试集
		:param t_v_t:[float,float,float] 分割所有数据train-validation-test的比例
		:param sfpath: 数据文件的名称，初始化时，会在数据根目录创建记录数据的csv文件，文件格式：label，*spectrum，如果已经有该记录文件
		:param shuffle: 是否将读取的数据打乱
		:param transform: 数据预处理/增强
		:param LoadCsvFile:根据数据存储格式自定义的读取文件数据的函数
		combined:生成器，第一个为光谱数据的header
		:param backEnd:str 存储文件的后缀
		:param supervised: 如果为无监督学习，将noising前的信号设置为label
		:param noising: callable——input：1d spectrum output：noised 1d spectrum

		"""

		# assert mode in ["train", "val", "test"]
		super(RamanDatasetCore, self).__init__()

		if t_v_t is None and k_split is None:  # 分割train-validation-test
			t_v_t = [0.7, 0.2, 0.1]
		# if type(t_v_t) is list:
		# 	t_v_t = numpy.array(t_v_t)

		self.k_split = k_split
		if k_split is not None:  # k_split 模式
			# t_v_t = [x*k_split for x in t_v_t]
			assert 0 <= k < k_split, "k must be in range [{},{}]".format(0, k_split - 1)
		self.k = k
		# assert t_v_t[0] + t_v_t[1] <= 1
		self.tvt = t_v_t
		self.new = newfile

		self.LoadCsvFile = LoadCsvFile
		self.root = dataroot

		self.name2label = {}  # 为每个分类创建一个label
		self.sfpath = sfpath
		self.shuff = shuffle
		self.mode = mode
		self.dataEnd = backEnd
		self.transform = transform
		self.unsupervised = unsupervised
		self.noising = noising

		self.train_split = self.tvt[0]
		self.validation_split = self.tvt[1]
		self.test_split = self.tvt[2]
		self.xs = None
		self.RamanFiles = []
		self.labels = []
		self.Ramans = []
		self.ratio = ratio
		if self.LoadCsvFile is None:
			self.LoadCsvFile = getRamanFromFile()

		for name in sorted(os.listdir(dataroot)):
			if not os.path.isdir(os.path.join(dataroot, name)):
				continue
			if not len(os.listdir(os.path.join(dataroot, name))):
				continue
			self.name2label[name] = len(self.name2label.keys())

		self.numclasses = len(self.name2label.keys())
		self.LoadCsv(sfpath)  # 加载所有的数据文件
		# 数据分割
		self.split_data()
		self.load_raman_data()  # 数据读取

	# def __add__(self, other):
	# 	if len(other) == 0:
	# 		return self
	# 	assert len(self.xs) == len(other.xs)
	# 	assert self.label2name().keys() == other.label2name().keys()
	# 	res = copy.deepcopy(self)
	# 	res.labels = other.labels + self.labels
	# 	res.Ramans += other.Ramans
	# 	res.RamanFiles += other.RamanFiles
	# 	return res

	def LoadCsv(self,
	            filename, ):
		pass

	def split_data(self):
		pass

	def load_raman_data(self):
		pass

	def shuffle(self):
		z = list(zip(self.Ramans, self.labels,self.RamanFiles))
		random.shuffle(z)
		self.Ramans[:], self.labels[:],self.RamanFiles = zip(*z)
		return

	def shufflecsv(self,
	               filename = None):
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

	def file2data(self):
		return dict(zip(self.RamanFiles, self.Ramans))

	def data2file(self):
		return dict(zip(self.Ramans, self.RamanFiles))

	def name2label(self):
		return self.name2label

	def label2name(self):
		keys = list(self.name2label.values())
		values = list(self.name2label.keys())
		return dict(zip(keys, values))

	def get_data_sorted_by_label(self):
		"""
		返回一个字典，每个label对应的数字指向一个tensor[label=label的光谱的数量,c=1,lenth]
		"""
		spectrum_each_label = {}
		for i in range(len(self.name2label)):
			spectrum_each_label[i] = None
		for i in range(self.__len__()):
			raman, label = self.Ramans[i], self.labels[i]
			raman = torch.unsqueeze(raman, dim = 0)

			if spectrum_each_label[label.item()] is None:
				spectrum_each_label[label.item()] = raman
			else:
				spectrum_each_label[label.item()] = torch.cat(
					(spectrum_each_label[label.item()],
					 raman),
					dim = 0
				)
		for k in spectrum_each_label.keys():
			if len(spectrum_each_label[k].shape) == 3:
				return spectrum_each_label
			else:
				spectrum_each_label[k] = pytorchlize(spectrum_each_label[k])
		return spectrum_each_label

	def wavelengths(self):
		return torch.tensor(self.xs)

	def show_data(self,
	              xs = None,
	              vis = None,
	              win = "data"):

		label2name = self.label2name()
		label2data = self.get_data_sorted_by_label()
		if xs == None:
			xs = self.wavelengths()
		if len(label2data[0].shape) > 2 and label2data[0].shape[-2] > 1:  # 多通道数据
			print("多通道数据")
			return

		if label2data[0].shape[-1] > 3:  # 光谱数据
			for i in range(self.numclasses):
				data = label2data[i]
				name = label2name[i]
				visdom_utils.spectrum_vis(data, win = win + "_" + name, xs = xs, vis = vis)
		else:  # 降维后数据
			# if vis == None:
			# 	if not win.endswith(".png"):
			# 		win += ".png"
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

	def savedata(self,
	             dir,
	             mode = "file_wise"):
		label2data = self.get_data_sorted_by_label()
		if not os.path.isdir(dir):
			os.makedirs(dir)
		if mode == "file_wise":
			for name in self.name2label.keys():
				path = os.path.join(dir, name + ".csv")
				data = label2data[self.name2label[name]]
				data = data.numpy()
				data = numpy.squeeze(data)
				numpy.savetxt(path, data, delimiter = ",")

	def __len__(self):
		return len(self.Ramans)

	def __getitem__(self,
	                item):
		assert item < len(self.Ramans), "{}/{}".format(item, len(self.Ramans))

		raman, label = self.Ramans[item], self.labels[item]

		if self.unsupervised:  # label设置为noising前的光谱数据
			label = raman.to(torch.float32)
			if not self.noising == None and self.mode == "train":
				raman = torch.squeeze(raman)
				raman = self.noising(raman)
				raman = torch.unsqueeze(raman, dim = 0).to(torch.float32)

		return raman, label


class Raman(RamanDatasetCore):
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
		super(Raman, self).__init__(*args, **kwargs)

	def LoadCsv(self,
	            filename, ):
		"""
		用于对数据分类并记录
		:param filename: 记录文件路径
		:return: 文件记录的所有lamen spectrum 和 label
		"""
		header = ["label", "filepath"]

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
		assert len(self.RamanFiles) == len(self.labels)
		return self.RamanFiles, self.labels  # [[float],[],...[]],[int]

	def split_data(self):
		train_split_int = int(self.train_split * len(self.RamanFiles))
		val_split_int = int((self.train_split + self.validation_split) * len(self.RamanFiles))

		if self.mode == "test":  # 分割测试集
			self.RamanFiles = self.RamanFiles[val_split_int:]
			self.labels = self.labels[val_split_int:]
			return

		if self.k_split is None:
			if self.mode == "train" and self.k_split is None:  # 分割训练集
				self.RamanFiles = self.RamanFiles[:train_split_int]
				self.labels = self.labels[:train_split_int]
			elif self.mode == "val" and self.k_split is None:  # 分割验证集
				self.RamanFiles = self.RamanFiles[train_split_int:val_split_int]
				self.labels = self.labels[train_split_int:val_split_int]
			else:
				raise Exception("Invalid mode!", self.mode)

		if self.k_split is not None:
			self.RamanFiles = self.RamanFiles[:val_split_int]
			self.labels = self.labels[:val_split_int]
			val_start = int(self.k / self.k_split * len(self.RamanFiles))
			val_end = int((self.k + 1) / self.k_split * len(self.RamanFiles))
			# l1 = self.RamanFiles[:val_start] + self.RamanFiles[val_end:]
			# l2 = self.RamanFiles = self.RamanFiles[val_start:val_end]
			# l = list(set(l1) & set(l2))
			if self.mode == "train":
				self.RamanFiles = self.RamanFiles[:val_start] + self.RamanFiles[val_end:]
				self.labels = self.labels[:val_start] + self.labels[val_end:]
			elif self.mode == "val":
				self.RamanFiles = self.RamanFiles[val_start:val_end]
				self.labels = self.labels[val_start:val_end]
		return self.RamanFiles,self.Ramans,self.labels
	def load_raman_data(self):
		self.Ramans = []
		for file in self.RamanFiles:
			R, X = self.LoadCsvFile(file)
			if self.xs is None:
				self.xs = X
			# 数据预处理
			assert R.shape[-1] == len(self.xs), "R:{},len_x:{},file:{}".format(R.shape, len(self.xs), file)

			if self.transform is not None:
				R[0] = self.transform(R[0])
			R = torch.tensor(R).to(torch.float32)
			self.Ramans.append(R)
		self.RamanFiles = [os.path.join(os.path.split(os.path.split(x)[0])[1], os.path.split(x)[1]) for x in
		                   self.RamanFiles]  # 只留分类和文件名

	def get_data_sorted_by_sample(self):
		return self.get_data_sorted_by_label()

	def sample2label(self):
		res = {}
		for i in range(self.numclasses):
			res[i] = i
		return res


class Raman_dirwise(RamanDatasetCore):
	def __init__(self,
	             *args,
	             **kwargs, ):
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
		if "sfpath" in kwargs.keys():
			kwargs["sfpath"] = kwargs["sfpath"][:-4]+"_dirwise"+kwargs["sfpath"][-4:]
		else:
			kwargs["sfpath"] = "Raman_dirwise.csv"
		super(Raman_dirwise, self).__init__(*args, **kwargs)
		self.spectrum_each_sample = None

	def LoadCsv(self,
	            filename, ):
		"""
		用于对数据分类并记录
		:param filename: 记录文件路径
		:return: 文件记录的所有lamen spectrum 和 label
		"""

		header = ["label", "dirpath", "files"]
		if not os.path.exists(os.path.join(self.root, filename)) or self.new:
			RamanFiles = []  # [dirpath,files]
			RamanDirs = []  # [dirpath]

			for name in self.name2label.keys():
				namepath = os.path.join(self.root, name, )
				l = os.listdir(namepath)
				for d in l:
					dirpath = os.path.join(namepath, d)
					if os.path.isdir(dirpath):  # 获取分类下所有文件夹
						RamanDirs.append(dirpath)
						files = []
						for f in os.listdir(dirpath):
							if f.endswith(self.dataEnd) and os.path.isfile(os.path.join(dirpath, f)):
								files.append(f)
						RamanFiles.append(files)

			if self.shuff:  # 打乱顺序
				z = list(zip(RamanDirs, RamanFiles))
				random.shuffle(z)
				RamanDirs, RamanFiles = zip(*z)

			assert len(RamanDirs) == len(RamanFiles)
			with open(os.path.join(self.root, filename), mode = "w", newline = "") as f:  # 记录所有数据
				writer = csv.writer(f)

				writer.writerow(header)
				for i in range(len((RamanDirs))):
					dir = RamanDirs[i]  # dir:data root/label name/sample dir
					files = RamanFiles[i]
					name = dir.split(os.sep)[-2]  # label name
					label = self.name2label[name]  # label idx
					writer.writerow([label, dir, *files])

		self.RamanFiles = []
		self.labels = []
		self.RamanDirs = []

		with open(os.path.join(self.root, filename)) as f:
			reader = csv.reader(f)
			for row in reader:
				if row == header:
					continue
				try:
					label = torch.tensor(int(row[0]))
					dir = row[1]
					files = row[2:]
					self.labels.append(label)
					self.RamanDirs.append(dir)
					self.RamanFiles.append(files)
				except:  # 数据格式有误
					print("wrong csv,remaking...")
					f.close()
					os.remove(os.path.join(self.root, filename))
					self.LoadCsv(filename)
					break
		assert len(self.RamanFiles) == len(self.labels)
		return self.RamanFiles, self.labels  # [[float],[],...[]],[int]

	def split_data(self):
		train_split_int = int(self.train_split * len(self.RamanFiles))
		val_split_int = int((self.train_split + self.validation_split) * len(self.RamanFiles))

		if self.mode == "test":  # 分割测试集
			self.RamanFiles = self.RamanFiles[val_split_int:]
			self.labels = self.labels[val_split_int:]
			self.RamanDirs = self.RamanDirs[val_split_int:]
			return

		if self.k_split is None:
			if self.mode == "train" and self.k_split is None:  # 分割训练集
				self.RamanFiles = self.RamanFiles[:train_split_int]
				self.RamanDirs = self.RamanDirs[:train_split_int]
				self.labels = self.labels[:train_split_int]
			elif self.mode == "val" and self.k_split is None:  # 分割验证集
				self.RamanFiles = self.RamanFiles[train_split_int:val_split_int]
				self.RamanDirs = self.RamanDirs[train_split_int:val_split_int]
				self.labels = self.labels[train_split_int:val_split_int]
			else:
				raise Exception("Invalid mode!", self.mode)

		if self.k_split is not None:
			self.RamanFiles = self.RamanFiles[:val_split_int]
			self.RamanDirs = self.RamanDirs[:val_split_int]
			self.labels = self.labels[:val_split_int]

			val_start = int(self.k / self.k_split * len(self.RamanFiles))
			val_end = int((self.k + 1) / self.k_split * len(self.RamanFiles))
			if self.mode == "train":
				self.RamanFiles = self.RamanFiles[:val_start] + self.RamanFiles[val_end:]
				self.RamanDirs = self.RamanDirs[:val_start] + self.RamanDirs[val_end:]
				self.labels = self.labels[:val_start] + self.labels[val_end:]
			elif self.mode == "val":
				self.RamanFiles = self.RamanFiles[val_start:val_end]
				self.RamanDirs = self.RamanDirs[val_start:val_end]
				self.labels = self.labels[val_start:val_end]

	def load_raman_data(self):
		RamanFiles = copy.deepcopy(self.RamanFiles)
		labels = copy.deepcopy(self.labels)
		RamanDirs = copy.deepcopy(self.RamanDirs)

		self.Ramans = []
		self.RamanFiles = []
		self.labels = []

		for i in range(len(RamanDirs)):
			dir = RamanDirs[i]
			files = RamanFiles[i]
			label = labels[i]
			for f in files:
				file = os.path.join(dir, f)
				R, X = self.LoadCsvFile(file)
				if self.xs is None:
					self.xs = X
				# 数据预处理
				assert R.shape[-1] == len(self.xs), "length:{} of the data from {} is invalid".format(R.shape[-1], file)

				if self.transform is not None:
					R[0] = self.transform(R[0])
				R = torch.tensor(R).to(torch.float32)
				self.Ramans.append(R)
				self.labels.append(label)
				self.RamanFiles.append(os.path.join(*file.split(os.sep)[-3:]))
		del self.RamanDirs

	def get_data_sorted_by_sample(self):
		# label2data = self.get_data_sorted_by_label()
		# labels : label Files:name/sample/datafile ramans:data
		if self.spectrum_each_sample is not None:
			return self.spectrum_each_sample
		spectrum_each_sample = {}
		for i in range(self.__len__()):
			raman, label = self.Ramans[i], self.labels[i]
			sample_num = self.RamanFiles[i].split(os.sep)[1]
			if not sample_num in spectrum_each_sample.keys():
				spectrum_each_sample[sample_num] = raman
			else:
				spectrum_each_sample[sample_num] = torch.cat(
					(spectrum_each_sample[sample_num],
					 raman),
					dim = 0
				)

		for k in spectrum_each_sample.keys():
			if len(spectrum_each_sample[k].shape) == 3:
				self.spectrum_each_sample = spectrum_each_sample
				return spectrum_each_sample
			else:
				spectrum_each_sample[k] = pytorchlize(spectrum_each_sample[k])
		self.spectrum_each_sample = spectrum_each_sample
		return spectrum_each_sample

	def sample2label(self):
		sample2label = {}
		for i in range(self.__len__()):
			label = self.labels[i]
			sample_num = self.RamanFiles[i].split(os.sep)[1]
			sample2label[sample_num] = label
		return sample2label

	def savedata(self,
	             dir,
	             mode = "file_wise",
	             simplyfied = True):
		if not os.path.isdir(dir):
			os.makedirs(dir)

		label2name = self.label2name()
		sample2label = self.sample2label()
		name2data = {}
		if simplyfied:
			sample2data = self.get_data_sorted_by_sample()
			for sample in sample2data.keys():
				data = sample2data[sample]
				label = sample2label[sample].item()
				name = label2name[label]
				data = data.numpy()
				data = numpy.squeeze(data).mean(axis = 0)
				data = numpy.expand_dims(data, axis = 0)
				name2data[name] = data if not name in name2data.keys() else numpy.vstack((name2data[name], data))
		else:
			label2data = self.get_data_sorted_by_label()
			for label in label2name.keys():
				name = label2name[label]
				name2data[name] = numpy.squeeze(label2data[label].numpy())

		X = self.xs
		for name in name2data.keys():
			data = name2data[name]

			if mode == "file_wise":
				path = os.path.join(dir, name + ".csv")
				numpy.savetxt(path, data, delimiter = ",")

			elif mode == "dir_wise":
				for i in range(len(data)):
					name_dir = os.path.join(dir, name)
					if not os.path.isdir(name_dir):
						os.makedirs(name_dir)

					path = os.path.join(name_dir, name + "-" + str(i) + ".csv")
					numpy.savetxt(path, data, delimiter = ",")
					numpy.savetxt(path, numpy.vstack((X, numpy.arange(len(X)), data[i])).T, delimiter = ",",
					              comments = "",
					              header = "Wavelength,Column,Intensity")

		return name2data


def pytorchlize(x):
	for i in range(3 - len(x.shape)):
		x = torch.unsqueeze(x, dim = len(x.shape) - 1)
	return x


if __name__ == '__main__':
	visdom_utils.startVisdomServer()
	raman = Raman
	readdatafunc0 = getRamanFromFile(wavelengthstart = 390, wavelengthend = 1810,
	                                 dataname2idx = {"Wavelength": 0, "Column": 2, "Intensity": 1}, )
	from scipy import interpolate


	def readdatafunc(
			filepath
	):
		R, X = readdatafunc0(filepath)
		R = numpy.squeeze(R)
		f = interpolate.interp1d(X, R, kind = "cubic")
		newX = numpy.linspace(400, 1800, 512)
		newR = f(newX)
		newR = numpy.expand_dims(newR, axis = 0)

		return newR, newX


	# ascconfig = dict(dataroot = os.path.join(projectroot, "data", "liver", "liver_all_basewise"),
	#                  LoadCsvFile = getRamanFromFile(wavelengthstart = 400, wavelengthend = 1800,
	#                                                 dataname2idx = {"Wavelength": 0, "Column": 1, "Intensity": 2},
	#                                                 delimeter = ",", ),
	#                  backEnd = ".csv", t_v_t = [1., 0., 0.], newfile = True)
	# db = Raman_dirwise(**ascconfig)
	# db.savedata(os.path.join(projectroot, "data", "liver_processed_mean"),mode = "dir_wise")

	dir = os.path.join(projectroot, "data", "liver_cell_dou")

	csvconfig_c = dict(dataroot = os.path.join(dir, "HepG2"),
	                   LoadCsvFile = readdatafunc,
	                   backEnd = ".csv", )
	csvconfig_h = dict(dataroot = os.path.join(dir, "MIHA"),
	                   LoadCsvFile = readdatafunc,
	                   backEnd = ".csv", )
	csvconfig_a = dict(dataroot = dir,
	                   LoadCsvFile = readdatafunc,
	                   backEnd = ".csv", t_v_t = [1.0,0,0])

	# db = Raman(dataroot = os.path.join(projectroot, "data", "zwf"),
	#                    LoadCsvFile = getRamanFromFile(wavelengthstart = 400, wavelengthend = 1800,
	#                                  dataname2idx = {"Wavelength": 5, "Column": 3, "Intensity": 4},
	#                                  delimeter = ",", ),
	#                    backEnd = ".csv", t_v_t = [1,0,0],shuffle = False)
	# header = "Wavenumber"
	# data_in = numpy.expand_dims(db.xs,axis = 0)
	# for i in range(len(db)):
	# 	file,data = db.RamanFiles[i],db.Ramans[i]
	# 	header+=","+file
	# 	data = data.numpy()
	# 	data_in = numpy.vstack((data_in,data))
	# numpy.savetxt("combined.csv",data_in.T,delimiter = ",",header = header)

	# db = raman(**csvconfig_c,
	#            newfile = True, t_v_t = [1, 0, 0])
	# db.show_data()
	# db = raman(**csvconfig_h,
	#            newfile = True, t_v_t = [1, 0, 0])
	# db.show_data()
	db = Raman_dirwise(**csvconfig_a,sfpath = "Raman_dirwise.csv",newfile = True,shuffle = False)
	# db.show_data(win = "all")
	db.savedata(os.path.join(os.path.dirname(dir),"liver_cell_filewise"),simplyfied = False)