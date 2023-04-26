import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import torch
import sys, os
import numpy as np
import copy

coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
projectroot = os.path.dirname(projectroot)
dataroot = os.path.join(projectroot, "data", )
sys.path.append(coderoot)

from bacteria.code_sjh.Core.basic_functions.data_functions import data2mean_std


def spectrum_vis_mpl(spectrums: torch.Tensor or numpy.ndarray,
                     xs = None,
                     name = None,
                     bias = 0,
                     side = True,
                     ax: plt.Axes = None,
                     line_color = None,
                     shadow_color = None,
                     title = None
                     ):
    """

    :param spectrums: a batch of spectrums[b,l] or [b,1,l]
    :param xs: the x-axis of the spectrum, default is None
    :param win: the window name in visdom localhost and the title of the window
    :param update: None if to rewrite and "append" if to add
    :param name: the name of the batch of spectrums, the line of the mean value is named as [name]+"_mean", mean + std
    -> [name] + "_up" , mean - std -> [name] + "_down"
    for example : If the name of the spectrums is "bacteria_R",then the names of the lines are "bacteria_R_mean",
    "bacteria_R_up" and "bacteria_R_down"
    :param opts: dict, {"main":dict(line visdom opts for the "mean" line), "side":dict(line visdom opts for the "up" line
     and "down" line)}
    :param bias: move the lines of spectrums up(positive number) or down(negative number)
    :param side: whether to show the mean + std and mean - std line
    :returns spectrum_mean ,spectrum_up=spectrum_mean+std,spectrum_down=spectrum_mean-std
    """

    if name is None:
        name = "spectrum"
    if shadow_color is None:
        shadow_color = "skyblue"
    if line_color is None:
        line_color = "blue"
    y_mean, y_up, y_down = data2mean_std(spectrums)
    y_mean += bias
    y_up += bias
    y_down += bias
    if xs is None:
        if type(spectrums) == torch.Tensor:
            xs = torch.arange(spectrums.shape[-1])
        elif type(spectrums) == numpy.ndarray:
            xs = numpy.arange(spectrums.shape[-1])

    assert xs.shape[-1] == y_mean.shape[-1], r"lenth of xs and spectrums doesn't fit"

    fig: plt.Figure()
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    ax.set_title(name if title is None else title)
    ax.set_xlabel("wavenumber cm$^{-1}$")
    ax.set_ylabel("intensity")
    ax.plot(xs, y_mean, label = name, color = line_color)
    if side:
        ax.fill_between(xs, y_down, y_up, color = shadow_color)

    return fig


if __name__ == '__main__':
    from bacteria.code_sjh.utils.RamanData import Raman_dirwise, Raman
    from bacteria.code_sjh.Core.basic_functions.fileReader import getRamanFromFile
    from scipy import interpolate
    from bacteria.code_sjh.utils import Process

    readdatafunc = getRamanFromFile(wavelengthstart = 200, wavelengthend = 2000,
                                    dataname2idx = {"Wavelength": 0, "Column": 2, "Intensity": 1}, )

    from pylab import mpl

    # 设置中文显示字体
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False

    dir = os.path.join(projectroot, "data", "tissue", "Try")
    csvconfig_a = dict(dataroot = dir,
                       LoadCsvFile = readdatafunc,
                       transform = Process.process_series([
                           Process.bg_removal_niter_fit(),
                           Process.interpolator(np.linspace(400, 1800, 512)),
                           Process.sg_filter(),
                           Process.norm_func()
                       ]),
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
