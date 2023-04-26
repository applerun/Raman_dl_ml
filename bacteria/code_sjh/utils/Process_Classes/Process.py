import copy
import glob
import os
import sys
import warnings

import numpy
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy import interpolate

from bacteria.code_sjh.Core.Preprocess import *
from bacteria.code_sjh.utils.Process_utils.baseline_remove import *

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
            warnings.warn("args not supported")
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
    def __init__(self, t = 0.01, ):
        super(noising_func_generator, self).__init__("noise(t={})".format(t))
        self.t = t

    def __call__(self, x, y = None):
        res = x + self.t * (x.mean()) * np.random.normal(0, 1, size = x.shape)
        if y is None:
            return res
        else:
            return res, y


def linearizeByterminal(slit: np.ndarray):
    shape = copy.deepcopy(slit.shape)
    l = len(slit.squeeze())
    for i in range(l):
        slit[i] = (l - i - 1) / (l - 1) * slit[0] + i / (l - 1) * slit[-1]
    return slit.reshape(shape)


class smoother(ProcessorFunction):
    def __init__(self, max_ = 1000,
                 span = 3, ):
        super(smoother, self).__init__("linearize_smoother(max_={},span={})".format(max_, span))
        self.max_ = max_
        self.span = span

    def __call__(self, y: np.ndarray, x = None):
        for i in range(len(y)):
            slit = y[max(0, i - self.span):min(len(y) - 1, i + self.span)]
            if y[i] - slit.min() > self.max_:
                y[max(0, i - self.span):min(len(y) - 1, i + self.span)] = linearizeByterminal(
                    y[max(0, i - self.span):min(len(y) - 1, i + self.span)])
        return y if x is None else (y, x)


class sg_filter(ProcessorFunction):
    def __init__(self, window_length = 11,
                 poly_order = 3, ):
        super(sg_filter, self).__init__("sg_filter(window_length={},poly_order={})".format(window_length, poly_order))
        self.window_length = window_length
        self.poly_order = poly_order

    def __call__(self, y, x = None):
        y = savgol_filter(y, self.window_length, self.poly_order)
        if x is None:
            return y
        else:
            return y, x


# spectral normalization
class norm_func(ProcessorFunction):
    def __init__(self, a = 0, b = 1, ):
        super(norm_func, self).__init__("norm" if (a == 0 and b == 1) else "norm({},{})".format(a, b))
        self.a = a
        self.b = b

    def __call__(self, y, x = None):
        y = ((self.b - self.a) * (y - min(y))) / (max(y) - min(y)) + self.a
        if x is None:
            return y
        else:
            return y, x


class area_norm_func(ProcessorFunction):
    def __init__(self, a = 1, ):
        assert a > 0, "a must be greater than 0"
        super(area_norm_func, self).__init__("area_norm" if a == 1 else "area_norm({})".format(a))
        self.a = a

    def __call__(self, y, x = None):
        y -= y.min()
        res = self.a * y / sum(y) * len(y)
        return res if x is None else (res, x)


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


def refile(src_file, readRaman, dst_file = None, xs = None, process = None):
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
    @param newfile: 是否覆盖旧的处理结果文件
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
    for dirpath, _, filenames in os.walk(dirname):
        for filename in filenames:
            if filename == savefilename:  # 被处理过
                file = os.path.join(dirpath, filename)
                os.remove(file)


nonfunc = ProcessorFunction

# delete_processed_walk()
# dir_process_walk()
