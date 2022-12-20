import copy
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

try:
    from baseline_remove import *

except:
    from bacteria.code_sjh.utils.Process_utils.baseline_remove import *
coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "data_ID")
sys.path.append(coderoot)

__all__ = ["sg_filter", "norm_func", "process_series", "noising_func_generator", "area_norm_func"]


# S-G filter
def noising_func_generator(t = 0.01):
    def func(x, y = None):
        res = x + t * (x.mean()) * np.random.normal(0, 1, size = x.shape)
        if y is None:
            return res
        else:
            return res, y

    return func


def sg_filter(window_length = 11,
              polyorder = 3):
    def func(x,y = None):
        x = savgol_filter(x, window_length, polyorder)
        if y is None:
            return x
        else:
            return x,y

    return func


# spectral normalization
def norm_func(a = 0,
              b = 1):
    def func(x,y = None):
        return ((b - a) * (x - min(x))) / (max(x) - min(x)) + a


    return func


def area_norm_func(a = 1):
    def func(x):
        return a * x / sum(x) * len(x)

    return func


# combined preprocessing function
def preprocess_default(x,
                       y = None):
    x = baseline_als(lam = 100000, p = 0.01, niter = 10)(x)
    # x = bg_removal_niter_piecewisefit()(x)
    # x = bg_removal_niter_fit()(x)
    x = sg_filter(window_length = 15, polyorder = 3)(x)
    x = norm_func(a = 0, b = 1)(x)
    if y == None:
        return x
    else:
        return x, y


def process_series(sequence,
                   copytype = "deepcopy"):
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


def refile(src_file, readRaman, process, dst_file = None, xs = None):
    raman, wavelength = readRaman(src_file)
    if raman[0][0] == 1:
        return
    if xs is not None:
        wavelength = xs
    if len(raman.shape) == 2:
        raman = raman.squeeze()
    raman = process(raman)
    if not os.path.isdir(os.path.dirname(dst_file)):
        os.makedirs(os.path.dirname(dst_file))

    if dst_file is None:
        dst_file = src_file
    if os.path.isfile(dst_file):
        os.remove(dst_file)
    np.savetxt(dst_file, np.vstack((wavelength, raman)).T, header = "Wavelength,Intensity", delimiter = ",",
               comments = "")


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


if __name__ == '__main__':
    def getRamanFromFile(wavelengthstart = 400,
                         wavelengthend = 1800,
                         dataname2idx = None,
                         delimeter = None):
        if wavelengthend < wavelengthstart:
            wavelengthstart, wavelengthend = wavelengthend, wavelengthstart
        dataname2idx = copy.deepcopy(dataname2idx)

        def func(filepath: str,
                 delimeter = delimeter, dataname2idx = dataname2idx):
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
                        print(filepath, ":", data, ",delimeter:", delimeter)
                        raise ValueError
                    if wavelengthstart < wavelength and wavelength < wavelengthend:
                        Ramans.append(intense)
                        Wavelengths.append(wavelength)
                    elif wavelength > wavelengthend:
                        break
            Ramans = np.array([Ramans])
            Wavelengths = np.array(Wavelengths)
            return Ramans, Wavelengths

        return func


    readdatafunc0 = getRamanFromFile(wavelengthstart = 390, wavelengthend = 1810,
                                     dataname2idx = {"Wavelength": 0, "Column": 2, "Intensity": 1}, )
    from scipy import interpolate


    def readdatafunc(
            filepath
    ):
        R, X = readdatafunc0(filepath)
        R = np.squeeze(R)
        f = interpolate.interp1d(X, R, kind = "cubic")
        newX = np.linspace(400, 1800, 512)
        newR = f(newX)
        newR = np.expand_dims(newR, axis = 0)

        return newR, newX


    src_dir = r"D:\myPrograms\pythonProject\Raman_dl_ml\bacteria\data\liver_cell_dou\MIHA"
    dst_dir = r".\data_res\MIHA"

    name2pre = {"bals": baseline_als(), "brnf": bg_removal_niter_fit(),
                "brnp": bg_removal_niter_piecewisefit()}
    for keys in name2pre.keys():
        baseline_remove = name2pre[keys]
        process = process_series([baseline_remove, sg_filter(), norm_func()])
        dst_dir_ = os.path.join(dst_dir, keys)
        dir_process_walk(src_dir, dst_dir_, readdata_func = readdatafunc, preprocess = process, newfile = True)

    # delete_processed_walk()
    # dir_process_walk()
