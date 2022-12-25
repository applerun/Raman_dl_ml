import copy
import sys, os

import numpy

coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "20211123")
sys.path.append(coderoot)

import torch
# torch.set_default_tensor_type(torch.FloatTensor)
import visdom
from torch import nn, optim
from torch.utils.data import DataLoader
try:
    from ..utils.RamanData import *
    from ..utils import Process, validation
except:
    from utils.RamenData import *
    from utils import Process, validation

from bacteria.code_sjh.Core.basic_functions.fileReader import getRamanFromFile

def show(spectrums, win, bias,std=True):
    y_mean, y_up, y_down = validation.data2mean_std(spectrums[:, 0, :])
    y_mean += bias
    y_down += bias
    y_up += bias
    lenth = spectrums.shape[-1]

    xs = torch.linspace(400, 1800, lenth)
    vis.line(X = xs, Y = y_mean,
             win = win,
             name = "label-" + label2name[bias] + "_mean",
             update = "append",
             opts = dict(title = win,
                         showlegend = True,
                         ))
    if std:
        vis.line(X = xs, Y = y_up,
                 win = win,
                 update = "append",
                 name = "label-" + label2name[bias] + "_up",
                 opts = dict(title = win,
                             showlegend = False,
                             dash = numpy.array(['dot']),
                             linecolor = numpy.array([[123, 104, 238]]),
                             ))

        vis.line(X = xs, Y = y_down,
                 win = win,
                 update = "append",
                 name = "label-" + label2name[bias] + "_down",
                 opts = dict(title = win,
                             showlegend = False,
                             dash = numpy.array(['dot']),
                             linecolor = numpy.array([[123, 104, 238]]),
                             ))


vis = visdom.Visdom()
# 准备预处理函数
sgf = Process.sg_filter()

nf = Process.norm_func()

bals = Process.baseline_als()  # 孙学长的预处理函数
brmi = Process.bg_removal_niter_fit()  # 黄志伟老师的预处理函数？
brnp = Process.bg_removal_niter_piecewisefit()  # 使用分段多项式拟合

t_v_t = [1, 0, .0]
train_db = Raman(dataroot, mode = "train",  backEnd = ".csv",
                 LoadCsvFile = getRamanFromFile(), t_v_t = t_v_t, transform = None)
label2name = train_db.label2name()

dic = train_db.get_data_sorted_by_label()
dic[4],dic[5]=dic[5],dic[4]
label2name[4],label2name[5] = label2name[5],label2name[4]
bias = 0
for bias in dic.keys():

    spectrums = dic[bias]
    num = spectrums.shape[0]
    processed_spectrums1 = copy.deepcopy(spectrums)
    processed_spectrums2 = copy.deepcopy(spectrums)
    for i in range(num):
        processed_spectrums1[i, 0, :] = torch.tensor(nf(sgf(bals(spectrums[i, 0, :].np()))))
        processed_spectrums2[i, 0, :] = torch.tensor(nf(sgf(brnp(spectrums[i, 0, :].np()))))
        spectrums[i, 0, :] = torch.tensor(nf(sgf(spectrums[i, 0, :].np())))

    show(spectrums, "raw", bias,std = False)
    show(processed_spectrums1, "bals", bias,std = False)
    show(processed_spectrums2, "brnp", bias,std = False)
