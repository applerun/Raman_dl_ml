import csv
import glob
import os
import random
import sys
import copy
import warnings

import numpy
import torch
import torch.utils.data

coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "data_ID")
sys.path.append(coderoot)
from bacteria.code_sjh.Core.basic_functions import visdom_func as visdom_utils
from bacteria.code_sjh.Core.RamanData import RamanDatasetCore, get_dict_str
from bacteria.code_sjh.Core.basic_functions.fileReader import getRamanFromFile
from bacteria.code_sjh.utils.RamanData.TensorFunctions import pytorchlize


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
				if type(self.ratio) == dict:
					if not name in self.ratio.keys():
						ratio = 1.0
					else:
						ratio = self.ratio[name]
					if ratio < 1.0:
						files = random.sample(files, int(ratio * len(files)))
					elif ratio > 1.0:
						import math
						int_ratio = int(math.floor(ratio))
						files = files * int_ratio + random.sample(files, int(ratio - int_ratio))
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
					warnings.warn("wrong csv,remaking...")
					f.close()
					os.remove(os.path.join(self.root, filename))
					return self.LoadCsv(filename)
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
			val_start = int(round(self.k * len(self.RamanFiles) / self.k_split))
			val_end = int(round((self.k + 1) * len(self.RamanFiles) / self.k_split))
			# l1 = self.RamanFiles[:val_start] + self.RamanFiles[val_end:]
			# l2 = self.RamanFiles = self.RamanFiles[val_start:val_end]
			# l = list(set(l1) & set(l2))
			if self.mode == "train":
				self.RamanFiles = self.RamanFiles[:val_start] + self.RamanFiles[val_end:]
				self.labels = self.labels[:val_start] + self.labels[val_end:]
			elif self.mode == "val":
				self.RamanFiles = self.RamanFiles[val_start:val_end]
				self.labels = self.labels[val_start:val_end]
		return self.RamanFiles, self.Ramans, self.labels

	def load_raman_data(self):
		self.Ramans = []
		for file in self.RamanFiles:
			R, X = self.LoadCsvFile(file)

			# 数据预处理
			if self.transform is not None:
				try:
					R, X = self.transform(R[0], X)
					R = numpy.expand_dims(R, 0)
				except:
					raise ValueError("data from {} cannot be transformed".format(file))
			R = torch.tensor(R).to(torch.float32)
			if self.xs is None:
				self.xs = X

			assert R.shape[-1] == len(self.xs), "R:{},len_x:{},file:{}".format(R.shape, len(self.xs), file)
			self.Ramans.append(R)
		self.RamanFiles = [x.replace(self.root + os.sep, "") for x in
		                   self.RamanFiles]  # 只留分类和文件名

	def get_data_sorted_by_sample(self):
		return self.get_data_sorted_by_label()

	def sample2label(self):
		res = {}
		for i in range(self.numclasses):
			res[i] = i
		return res


def Raman_depth_gen(max_depth,
                    min_depth = None,
                    warning = False):  # 生成一个Raman的子类的walk版本，只读取目录下指定深度的文件
	assert max_depth > 0, "max_depth should not be less than 1"
	if min_depth is None:
		min_depth = 0
	assert min_depth <= max_depth, "max_depth should not be less than min_depth"

	class Raman_t(Raman):
		"""
		通过参数 max_depth,min_depth 指定遍历深度
		只读取直属于每个分类文件夹下时max_depth = 1 效果等同于 Raman
		也可通过参数 min_depth
		"""

		def __init__(self,
		             *args,
		             **kwargs):
			"""
			Raman 的进阶版，遍历一个文件夹下的所有文件
			通过参数 max_depth 指定遍历深度
			只读取直属于每个分类文件夹下时max_depth = 1 效果等同于 Raman
			也可通过参数 min_depth
			"""
			super(Raman_t, self).__init__(*args, **kwargs)
			self.sfpath = self.sfpath + "_depth_{}_to_{}".format(min_depth, max_depth)

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
				RamanLabels = []
				for name in self.name2label.keys():
					files = []

					for r, _, f in os.walk(os.path.join(self.root, name), ):
						d = len(r.replace(self.root, "").split(os.sep)) - 1
						if d < min_depth:
							if warning:
								print("depth {} < min_depth {}, ignore dir {}".format(
									d, max_depth, r))
							continue
						elif d > max_depth:
							if warning:
								print("depth {} > max_depth {}, ignore dir {} ,...".format(
									d, max_depth, r))
							continue
						f = list(filter(lambda x: x.endswith(self.dataEnd), f))
						files += [os.path.join(r, x) for x in f]

					# files = glob.glob(os.path.join(self.root, name, "*" + self.dataEnd))
					if type(self.ratio) == dict:
						if not name in self.ratio.keys():
							ratio = 1.0
						else:
							ratio = self.ratio[name]
						if ratio < 1.0:
							files = random.sample(files, int(ratio * len(files)))
						elif ratio > 1.0:
							import math
							int_ratio = int(math.floor(ratio))
							files = files * int_ratio + random.sample(files, int(ratio - int_ratio))

					RamanFiles += files
					RamanLabels += [self.name2label[name]] * len(files)

				if self.shuff:  # 打乱顺序
					z = list(zip(RamanFiles, RamanLabels))
					random.shuffle(z)
					RamanFiles, RamanLabels = zip(*z)

				with open(os.path.join(self.root, filename), mode = "w", newline = "") as f:  # 记录所有数据
					writer = csv.writer(f)

					writer.writerow(header)
					for i in range(len(RamanFiles)):
						spectrum = RamanFiles[i]
						label = RamanLabels[i]
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

	return Raman_t


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
			kwargs["sfpath"] = kwargs["sfpath"][:-4] + "_dirwise" + kwargs["sfpath"][-4:]
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

				if self.transform is not None:
					R, X = self.transform(R[0], X)
					R = numpy.expand_dims(R, 0)
				if self.xs is None:
					self.xs = X
				# 数据预处理
				assert R.shape[-1] == len(self.xs), "length:{} of the data from {} is invalid".format(R.shape[-1], file)
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
			sample_num = os.path.split(self.RamanFiles[i])[0]
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
			sample_num = os.path.split(self.RamanFiles[i])[0]
			sample2label[sample_num] = label
		return sample2label

	def savedata(self,
	             dir,
	             mode = "file_wise",
	             backend = ".csv",
	             delimiter = ",",
	             simplyfied = True):
		"""
		TODO: 生成文件的时候能按照sample再生成一级文件夹
		@param dir: 保存的目标文件夹
		@param mode: "file_wise":一类的所有光谱保存到同一个文件中；"dir_wise"：光谱按照类别保存，一个光谱一个文件
		@param backend: 保存文件后缀
		@param delimiter: 数据分割符
		@param simplyfied: True按照文件夹保存（sample文件夹内所有光谱取均值） 所有光谱都保存
		@return:
		"""
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
				path = os.path.join(dir, name + backend)
				numpy.savetxt(path, data, delimiter = ",")

			elif mode == "dir_wise":
				for i in range(len(data)):
					name_dir = os.path.join(dir, name)
					if not os.path.isdir(name_dir):
						os.makedirs(name_dir)

					path = os.path.join(name_dir, name + "-" + str(i) + backend)
					# numpy.savetxt(path, data, delimiter = ",")
					numpy.savetxt(path, numpy.vstack((X, numpy.arange(len(X)), data[i])).T, delimiter = delimiter,
					              comments = "",
					              header = "Wavelength,Column,Intensity")

		return name2data


# class Raman_dirwise_labelfile(RamanDatasetCore)


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
	                   backEnd = ".csv", t_v_t = [1.0, 0, 0])

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
	db = Raman_dirwise(**csvconfig_a, sfpath = "Raman_dirwise.csv", newfile = True, shuffle = False)
	# db.show_data(win = "all")
	db.savedata(os.path.join(os.path.dirname(dir), "liver_cell_filewise"), simplyfied = False)
