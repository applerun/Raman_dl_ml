"""
读取
"""

import os.path
import numpy
import torch
from bacteria.code_sjh.utils import Process
from bacteria.code_sjh.Core.basic_functions import fileReader
from bacteria.code_sjh.utils.RamanData import Raman, Raman_dirwise
from bacteria.code_sjh.utils.Process_utils import baseline_remove
from bacteria.code_sjh.Core.basic_functions import mpl_func
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba_array, to_rgba

colors = list(mcolors.XKCD_COLORS.keys())

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'

project_root = r".."
data_root = os.path.join(project_root, "Sample_data", "Sample_bad_signal")

print(os.path.abspath(data_root))
print(os.path.isdir(data_root))
readdatafunc = fileReader.getRamanFromFile(-200, 4000, dataname2idx = {"Wavelength": 0, "Intensity": 1},
                                           delimeter = "\t")
db_cfg = dict(
    backEnd = ".asc",
    # backEnd = ".asc",
    LoadCsvFile = readdatafunc,
    k_split = 6,
    transform =
    Process.process_series([  # 设置预处理流程
        Process.interpolator(numpy.linspace(0, 3400, 1024)),  # 读取的时候预先插值
    ]
    )
)

db_glass = Raman(dataroot = os.path.join(data_root, "glass"), **db_cfg, mode = "all")
db_tissue = Raman_dirwise(dataroot = os.path.join(data_root, "raw_data"), **db_cfg, mode = "all")

glassdata = db_glass.get_data_sorted_by_label()[0].numpy()

fig = mpl_func.spectrum_vis_mpl(glassdata, db_glass.xs, name = "glass")
glass_mean, _, _ = mpl_func.data2mean_std(glassdata)

# 多项式拟合
glass_p = numpy.polyfit(db_glass.xs, glass_mean, deg = 9)
glass_p = numpy.poly1d(glass_p)(db_tissue.xs)

mpl_func.spectrum_vis_mpl(glass_p, db_tissue.xs, name = "glass_p", ax = fig.axes[0], side = False)


# baselineremove = baseline_remove.baseline_als(lam = 10000, p = 0.01, niter = 10)


def plot_process(svdir = os.path.join(project_root, "Sample_results", "Sample_bad_signal", "plot"),
                 pointnum = 512,  # 插值点数
                 lam = 10000, p = 0.01, niter = 10,  # bals
                 windor_length = 11, poly_order = 3,  # sg
                 newX = True,  # 是否要插值
                 pltsave_in_one = False,  # 是否保存所有处理后结果在一张图中
                 substrate_glass = True,  # 是否减玻璃背景
                 tissue_wise = False,  # True :每个tissue 单独画图，samplewise ：每个label单独画图
                 ):
    if tissue_wise:
        name2data_tissue = db_tissue.get_data_sorted_by_sample()  # 按照编号画图
    else:
        name2data_tissue = db_tissue.get_data_sorted_by_name()  # 选择是按照信号强弱绘图

    newX = numpy.linspace(450, 1800, pointnum) if newX else None
    series = [
        # baseline_remove.baseline_als(lam = lam, p = p, niter = niter),
        baseline_remove.airALS(lam = lam,p = p,niter = niter),
        Process.interpolator(newX) if newX is not None else Process.none_func,
        Process.sg_filter(window_length = windor_length, poly_order = poly_order),
        Process.area_norm_func(),
    ]
    data_process_func = Process.process_series(series)

    data_process_func = Process.batch_process(data_process_func, verbose = True)
    bias = 0
    if pltsave_in_one:
        fig, axes = plt.subplots(1, 1)

    for name in name2data_tissue.keys():
        tissue_data = name2data_tissue[name].numpy()
        # 减玻璃背景
        if substrate_glass:
            fig2 = mpl_func.spectrum_vis_mpl(tissue_data, db_tissue.xs, name = name + "ori", title = name)
            tissue_data = tissue_data - glass_p
            mpl_func.spectrum_vis_mpl(tissue_data, db_tissue.xs, name = name + "after", ax = fig2.axes[0], )
            fig2.axes[0].legend()

        if newX is not None:
            tissue_data, _ = data_process_func(tissue_data, db_tissue.xs)
        else:
            tissue_data = data_process_func(tissue_data)
        c = mcolors.XKCD_COLORS[colors.pop()]
        c = to_rgba(c, 1)

        if not pltsave_in_one:
            fig, axes = plt.subplots(1, 1)

        mpl_func.spectrum_vis_mpl(tissue_data, db_tissue.xs if newX is None else newX, name = name, ax = axes,
                                  bias = bias,
                                  title = "processed",
                                  line_color = to_rgba(c, 1), shadow_color = to_rgba(c, 0.6)
                                  )
        bias += tissue_data.max()
        if not pltsave_in_one:
            sv_file = os.path.join(svdir, name + ".png")
            dir = os.path.dirname(sv_file)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            fig.savefig(sv_file)
            plt.close(fig)
    if pltsave_in_one:
        sv_file = os.path.join(svdir, ("tissue_wise" if tissue_wise else "sampel_wise")+"_all.png")
        dir = os.path.dirname(sv_file)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        axes.legend()
        fig.savefig(sv_file)
    if substrate_glass:
        plt.show()


lambdas = range(100, )
if __name__ == '__main__':
    plot_process(svdir = os.path.join(project_root, "Sample_results", "Sample_bad_signal", "plot_airALS"),
                 pointnum = 512,  # 插值点数
                 lam = 10000, p = 0.01, niter = 10,  # bals
                 windor_length = 11, poly_order = 3,  # sg
                 newX = True,  # 是否要插值
                 pltsave_in_one = True,  # 是否保存所有处理后结果在一张图中
                 substrate_glass = False,
                 tissue_wise = True)
