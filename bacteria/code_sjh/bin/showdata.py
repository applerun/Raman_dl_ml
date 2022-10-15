import numpy as np
import torch

from bacteria.code_sjh.utils import Process, RamanData
from bacteria.code_sjh.utils.Validation.visdom_utils import data2mean_std
from bacteria.code_sjh.utils.Validation.mpl_utils import spectrum_vis_mpl
from conv_net_classify import readdatafunc
import os, time
import matplotlib.pyplot as plt

projectroot = __file__
for i in range(4):
    projectroot = os.path.dirname(projectroot)
dataroot = os.path.join(projectroot, "data", "tissue", "Try")

raman = RamanData.Raman

db_cfg = dict(  # 数据集设置
    dataroot = dataroot,
    backEnd = ".csv",
    # backEnd = ".asc",
    t_v_t = [1.0, 0.0, 0.0],
    LoadCsvFile = readdatafunc,
    k_split = 6,
    transform = Process.process_series([  # 设置预处理流程
        # Process.baseline_als(),
        # Process.bg_removal_niter_fit(),
        Process.bg_removal_niter_piecewisefit(),
        Process.sg_filter(),
        Process.norm_func(), ]
    ))



if __name__ == '__main__':
    from bacteria.code_sjh.utils.RamanData import Raman_dirwise, getRamanFromFile, Raman
    from scipy import interpolate

    readdatafunc0 = getRamanFromFile(wavelengthstart = 390, wavelengthend = 1810,
                                     dataname2idx = {"Wavelength": 0, "Column": 2, "Intensity": 1}, )

    from pylab import mpl

    # 设置中文显示字体
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    def readdatafunc(  # 插值，将光谱长度统一为512
            filepath
    ):
        R, X = readdatafunc0(filepath)
        R = np.squeeze(R)
        f = interpolate.interp1d(X, R, kind = "cubic")
        newX = np.linspace(400, 1800, 512)
        newR = f(newX)
        newR = np.expand_dims(newR, axis = 0)
        return newR, newX


    dir = os.path.join(projectroot, "bacteria","data", "tissue", "Try")
    csvconfig_a = dict(dataroot = dir,
                       LoadCsvFile = readdatafunc,
                       backEnd = ".csv", t_v_t = [1, 0, 0])
    # raman = Raman_dirwise
    raman = Raman
    db = raman(**csvconfig_a, sfpath = "Raman_dirwise.csv", newfile = True, shuffle = False)
    # db = Raman(**csvconfig_a, )
    label2data = db.get_data_sorted_by_label()
    label2name = db.label2name()
    fig, ax = plt.subplots(len(list(label2data.keys())), dpi = 200)
    for label in label2data.keys():
        data = label2data[label].numpy()
        name = label2name[label]
        spectrum_vis_mpl(data, db.xs, ax = ax[label], name = name)
    plt.subplots_adjust(wspace = 0.25)
    plt.show()
    db.show_data()