import copy
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
try:
	from .Process_utils.baseline_remove import *
except:
	from bacteria.code_sjh.utils.Process_utils.baseline_remove import *
coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "data_ID")
sys.path.append(coderoot)




__all__ = ["sg_filter","norm_func","process_series", "noising_func_generator"]
# S-G filter
def noising_func_generator(t = 0.01):
	def func(x):
		res = x + t * (x.mean()) * np.random.normal(0, 1, size = x.shape)
		return res

	return func

def sg_filter(window_length = 11, polyorder = 3):
	def func(x):
		x = savgol_filter(x, window_length, polyorder)
		return x

	return func

# spectral normalization
def norm_func(a = 0, b = 1):
	def func(x):
		return ((b - a) * (x - min(x))) / (max(x) - min(x)) + a

	return func


# combined preprocessing function
def preprocess_default(x, y = None):
	x = baseline_als(lam = 100000, p = 0.01, niter = 10)(x)
	# x = bg_removal_niter_piecewisefit()(x)
	# x = bg_removal_niter_fit()(x)
	x = sg_filter(window_length = 15, polyorder = 3)(x)
	x = norm_func(a = 0, b = 1)(x)
	return x


def process_series(sequence, copytype = "deepcopy"):
	ctype2cfunc = {"copy": copy.copy, "deepcopy": copy.deepcopy}

	# 将所有处理函数结合在一起
	def func(x):
		y = ctype2cfunc[copytype](x)
		for funcs in sequence:
			y = funcs(y)
		return y

	return func


def dir_process(dirname = "**",
                datafilename = "combined-.csv",
                savefilename = "combined-p.csv",
                preprocess = preprocess_default,
                dataroot = dataroot):
	# 原本process.py文件的功能
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


def dir_process_walk(dirname = dataroot,
                     datafilename = "combined-.csv",
                     savefilename = "combined-p.csv",
                     preprocess = preprocess_default):
	# 将dirname下面所有名为datafilename的文件process并保存为savefilename文件
	for dirpath, _, filenames in os.walk(dirname):

		for filename in filenames:
			if savefilename not in filenames and filename == datafilename:  # 没有被处理过
				file = os.path.join(dirpath, filename)
				df = pd.read_csv(file)
				print(dirpath, ":done")
				# data = preprocess(pd.read_csv(file).values)

				m = df.shape[0]  # number of lines
				n = df.shape[1]  # number of columns
				matrix = np.ndarray(shape = (m, n))

				for i in range(1, m + 1, 1):
					data = df.iloc[i - 1]
					data_p = preprocess(data)
					matrix[i - 1, :] = data_p

				savefilepath = os.path.join(os.path.dirname(file), savefilename)
				np.savetxt(savefilepath, matrix, delimiter = ',')  # save the matrix into a new csv file


def delete_processed_walk(dirname = dataroot, savefilename = "combined-p.csv"):
	for dirpath, _, filenames in os.walk(dirname):
		for filename in filenames:
			if filename == savefilename:  # 被处理过
				file = os.path.join(dirpath, filename)
				os.remove(file)


if __name__ == '__main__':
	delete_processed_walk()
	dir_process_walk()
