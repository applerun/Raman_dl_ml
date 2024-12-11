import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

from scipy import interpolate

try:
	from PreProcessor import *
except:
	from .PreProcessor import *

coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "data_ID")
sys.path.append(coderoot)


# __all__ = ["interpolator", "noising_func_generator", "smoother", "sg_filter", "norm_func", "area_norm_func", "nonfunc"]


class interpolator(ProcessorFunction):
	def __init__(self, *args, **kwargs):
		"""
		@param args:
			1.newX:ndarray
				newX = newX
			2.a:float,b:float,n:int
				newX = np.linspace(a,b,n)
		@param kwargs:
			newX:新的横坐标
		"""
		if "newX" in kwargs.keys():
			newX = kwargs["newX"]
		elif len(args) == 1:
			newX = args[0]
		elif len(args) == 3:
			newX = numpy.linspace(*args)
		else:
			warnings.warn("args not supported:{}".format(args))
			newX = numpy.linspace(0, 1, 512)

		super(interpolator, self).__init__("interpolator({},{},{})".format(newX.min(), newX.max(), len(newX)))
		self.newX = newX

	def __call__(self, y, x = None):
		if x is None:
			warnings.warn("Need x specified, not None")
		f = interpolate.interp1d(x, y, kind = "cubic")
		new_y = f(self.newX)
		return new_y, self.newX


class noising_func_generator(ProcessorFunction):
	# 添加噪声（用于自监督学习）
	def __init__(self, t = 0.01, ):
		super(noising_func_generator, self).__init__("noise(t={})".format(t))
		self.t = t

	def __call__(self, x, y = None):
		res = x + self.t * (x.mean()) * np.random.normal(0, 1, size = x.shape)
		if y is None:
			return res
		else:
			return res, y


# spectral normalization


# combined preprocessing function


preprocess_default = ProcessorRootSeries([
	DeepCopy(),
	baseline_als(),
	sg_filter(),
	norm_func(),
])


def dir_process(dirname = "**",
				datafilename = "combined-.csv",
				savefilename = "combined-p.csv",
				preprocess = preprocess_default,
				dataroot = dataroot):
	# 遍历所有符合os.path.join(dataroot, dirname, datafilename)匹配的文件
	# 将其中数据作为光谱用preprocess处理后，保存在os.path.join(dataroot, dirname, savefilename)

	files = glob.glob(os.path.join(dataroot, dirname, datafilename))
	for file in files:
		df = pd.read_csv(file)

		# data = preprocess(pd.read_csv(file).values)

		m = df.shape[0]  # number of lines
		n = df.shape[1]  # number of columns
		# print('number of lines:', m)
		# print('number of columns:', n)

		matrix = np.ndarray(shape = (m, n))

		for i in range(1, m + 1, 1):
			data = df.iloc[i - 1]
			data_p = preprocess(data)
			matrix[i - 1, :] = data_p
			print('line', i - 1, 'completed')
		savefilepath = os.path.join(os.path.dirname(file), savefilename)
		np.savetxt(savefilepath, matrix, delimiter = ',')  # save the matrix into a new csv file


def refile(src_file, readRaman, dst_file = None, xs = None, process = None):
	# 读取一个光谱，并保存为dstfile（如果没有定义dstfile则覆盖原文件
	raman, wavelength = readRaman(src_file)
	if raman[0][0] == 1:
		return
	if xs is not None:
		wavelength = xs
	if len(raman.shape) == 2:
		raman = raman.squeeze()
	if process is not None:
		raman = process(raman)
	if not os.path.isdir(os.path.dirname(dst_file)):
		os.makedirs(os.path.dirname(dst_file))

	if dst_file is None:
		dst_file = src_file
	if os.path.isfile(dst_file):
		os.remove(dst_file)
	np.savetxt(dst_file, np.vstack((wavelength, raman)).T, header = "Wavelength,Intensity", delimiter = ",",
			   comments = "", fmt = "%s")


def dir_process_walk(src_dir,
					 dst_dir,
					 readdata_func = None,
					 preprocess = preprocess_default,
					 newfile = True
					 ):
	"""

	@param src_dir:
	@param src_dir: 旧文件夹
	@param dst_dir: 新文件夹
	@param readdata_func: 数据读取函数
	@param preprocess: 预处理函数
	@param newfile: True ：如果存在dst dir且文件夹下需要覆盖旧的处理结果文件，则覆盖
	@return:
	"""
	# 将dirname下面所有名为datafilename的文件process并保存为savefilename文件
	for dirpath, _, filenames in os.walk(src_dir):
		dst_subdir = dirpath.replace(dirpath, dst_dir)
		for filename in filenames:
			dst_file = os.path.join(dst_subdir, filename)
			src_file = os.path.join(dirpath, filename)
			if os.path.isfile(dst_file) and newfile is False:
				continue
			try:
				refile(src_file, readdata_func, preprocess, dst_file)
			except:
				print(src_file, " failed")


# if dst_dir not in filenames and filename == src_dir:  # 没有被处理过
#     file = os.path.join(dirpath, filename)
#     df = pd.read_csv(file)
#     print(dirpath, ":done")
#     # data = preprocess(pd.read_csv(file).values)
#
#     m = df.shape[0]  # number of lines
#     n = df.shape[1]  # number of columns
#     matrix = np.ndarray(shape = (m, n))
#
#     for i in range(1, m + 1, 1):
#         data = df.iloc[i - 1]
#         data_p = preprocess(data)
#         matrix[i - 1, :] = data_p
#
#     savefilepath = os.path.join(os.path.dirname(file), dst_dir)
#     np.savetxt(savefilepath, matrix, delimiter = ',')  # save the matrix into a new csv file


def delete_processed_walk(dirname = dataroot,
						  savefilename = "combined-p.csv"):
	# 删除所有文件名为savefilename的文件（通常为预处理后的"combined-p.csv"）
	for dirpath, _, filenames in os.walk(dirname):
		for filename in filenames:
			if filename == savefilename:  # 被处理过
				file = os.path.join(dirpath, filename)
				os.remove(file)


def str2processor(series: str, prefix = ""):
	# 将字符串转换为一个Processor实例，可以理解为str(Processor)的逆操作
	# 使用场景：将str(Processor)保存在文本文件中，下次使用可以直接利用此函数+文本文件读取构建Processor
	list_processor_str = series.split(" --> ")
	if not prefix.endswith(".") and len(prefix) > 0:
		prefix = prefix + "."
	list_processor = [eval(prefix + x) for x in list_processor_str]
	if len(list_processor) == 0:
		return ProcessorRoot()
	elif len(list_processor) == 1:
		return list_processor[0]
	else:
		return ProcessorRootSeries(list_processor)


none_func = ProcessorFunction  # 不作任何操作的预处理步骤
process_series = ProcessorRootSeries
