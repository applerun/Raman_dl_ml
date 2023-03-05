import matplotlib.pyplot as plt
import torch
import sys, os
from bacteria.code_sjh.utils.RamanData import Raman_dirwise, Raman
from bacteria.code_sjh.Core.basic_functions.fileReader import getRamanFromFile
from bacteria.code_sjh.Core.basic_functions.path_func import getRootPath
from bacteria.code_sjh.Core.basic_functions.mpl_func import spectrum_vis_mpl
from bacteria.code_sjh.Core.RamanData import RamanDatasetCore
from bacteria.code_sjh.utils import Process
from pylab import mpl

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
projectroot = getRootPath("bacteria")
readdatafunc = getRamanFromFile(wavelengthstart = 390, wavelengthend = 1810,
                                dataname2idx = {"Wavelength": 0, "Column": 2, "Intensity": 1}, )
import numpy as np


# 设置中文显示字体


def db_plot_samplewise(db: RamanDatasetCore, dst, backend = ".png", ):
    sample2data = db.get_data_sorted_by_sample()
    for sample in sample2data.keys():
        data = sample2data[sample]
        fig, ax = plt.subplots(1, dpi = 200)
        spectrum_vis_mpl(data, db.xs, ax = ax, name = sample)
        dstfile = os.path.join(dst, sample + backend)
        if not os.path.isdir(os.path.dirname(dstfile)):
            os.makedirs(os.path.dirname(dstfile))
        plt.savefig(dstfile)
        plt.close(fig)


if __name__ == '__main__':
    # plot所有mol分子分型的光谱
    mol = "1p19q(缺-1)"
    dataroot_batch1 = os.path.join(projectroot, "data", "脑胶质瘤", "data_classified")
    dataroot_batch2 = os.path.join(projectroot, "data", "脑胶质瘤", "data_indep")
    dstroot = os.path.join(projectroot, "data", "脑胶质瘤", "data_plot")
    csvconfig_a = dict(dataroot = os.path.join(dataroot_batch1, mol),
                       LoadCsvFile = readdatafunc,
                       transform = Process.process_series([
                           Process.bg_removal_niter_fit(),
                           Process.intorpolator(np.linspace(400, 1800, 512)),
                           Process.sg_filter(),
                           Process.norm_func()
                       ]),
                       backEnd = ".csv", t_v_t = [1, 0, 0])
    db1 = Raman_dirwise(**csvconfig_a, newfile = False, shuffle = False, sfpath = "Raman_dirwise.csv")
    csvconfig_a["dataroot"] = os.path.join(dataroot_batch2, mol)
    db2 = Raman_dirwise(**csvconfig_a, newfile = False, shuffle = False, sfpath = "Raman_dirwise.csv")
    db_plot_samplewise(db1, dstroot, "_old.png")
    print("batch1 ploted")
    db_plot_samplewise(db2, dstroot, "_new.png")
    print("batch2 ploted")