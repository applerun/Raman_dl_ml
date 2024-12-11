import numpy as np
import scipy.signal
from bacteria.code_sjh.utils.RamanData import Raman_dirwise
from bacteria.code_sjh.utils.Process_utils import Process


def msc(X: np.ndarray, Au: np.ndarray):
    assert X.shape[-1] == Au.shape[-1]
    me = np.mean(X, axis = 0)
    # me = np.mean(np.vstack((np.expand_dims(me, axis = 0), Au)), axis = 0)
    m, p = np.shape(X)
    X_msc = np.zeros((m, p))

    for i in range(m):
        poly = np.polyfit(me, X[i], 1)  # 每个样本做一次一元线性回归
        for j in range(p):
            X_msc[i, j] = (X[i, j] - poly[1]) / poly[0]
    # poly = np.polyfit(me, Au[0], 1)
    # Au_msc = np.zeros(p)
    # for j in range(p):
    #     Au_msc[j] = Au[0][j] / [poly[0]] - poly[1] / poly[0]
    Au_msc = Au[0]
    return X_msc, Au_msc


class Raman_Auremoval(Raman_dirwise):
    def __init__(self, *arg, **kwargs):
        super(Raman_Auremoval, self).__init__(*arg, **kwargs)


def niter_removal(R, X, Au,
                  preprocess = Process.process_series([
                      # Process.bg_removal_niter_fit(num_iter = 100),
                      Process.baseline_als(),
                      Process.sg_filter(window_length = 17),
                      Process.norm_func()
                  ])):
    R_n = np.squeeze(R.np())
    for i in range(R_n.shape[0]):
        R_n[i] = preprocess(R_n[i])
        # R_p = Process.preprocess_default(R_msc[i])
        # # w = (R_p > np.squeeze(Au_p)) * p + (R_p < np.squeeze(Au_p)) * (1 - p)
        # w = 1
        # AuProcessed[i] = w*R_p-Au_p
    Au_n = preprocess(Au[0])
    R_n -= Au_n
    res = np.zeros_like(R_n)
    for i in range(res.shape[0]):
        res[i] = Process.norm_func()(R_n[i])
    return res


def msc_removal(R, X, Au, l = 0.7, p = 0.5):
    R_msc, Au_msc = msc(np.squeeze(R.np()), Au)  # 多元散射矫正

    # 缩放Au信号，削除Si峰
    Peaks_Au = scipy.signal.find_peaks(Au_msc,
                                       # height = Au_msc.max() * l + Au_msc.min() * (1 - l),
                                       prominence = 1,
                                       )
    Peaks_R = scipy.signal.find_peaks(R_msc.mean(axis = 0),
                                      # height = R_msc.max() * l + R_msc.min() * (1 - l),
                                      prominence = 1,
                                      )
    Au_msc *= Peaks_R[1]["prominences"].max() / Peaks_Au[1]["prominences"].max()

    w = (R_msc > np.squeeze(Au_msc)) * p + (R_msc < np.squeeze(Au_msc)) * (1 - p)
    AuRemoved = (R_msc - np.squeeze(Au_msc)) * w
    return AuRemoved


def AuRemove(db: Raman_dirwise, func, Au_src = None, readdata = None, R_dst = None):
    """
    db:数据集
    Au_src:金基底数据
    R_dst:输出数据
    """
    if Au_src is None:
        Au_src = os.path.join(projectroot, "data", "liver", "Au")
    if R_dst is None:
        R_dst = os.path.join(os.path.dirname(db.root), os.path.split(db.root)[1] + "_Au_removed")
    if readdata is None:
        readdata = db.LoadCsvFile
    sample2label = db.sample2label()
    label2name = db.label2name()

    base2data = db.get_data_sorted_by_sample()

    for key in base2data.keys():
        label = sample2label[key].item()
        name = label2name[label]
        dst = os.path.join(R_dst, name, key)
        file_without_backend = key.split("-")[-1]

        R = base2data[key]
        Au, X = readdata(os.path.join(Au_src, file_without_backend + ".csv"))
        if not os.path.isdir(dst):
            os.makedirs(dst)
        try:
            R_new = func(R, X, Au)
        except:
            print(os.path.join(Au_src, file_without_backend + ".csv"))
            raise AssertionError
        for i in range(len(R)):
            filename = name + "-" + str(i) + ".csv"
            filepath = os.path.join(dst, filename)
            try:
                np.savetxt(filepath, np.vstack((X, np.arange(len(X)), R_new[i])).T, delimiter = ",",
                           comments = "",
                           header = "Wavelength,Column,Intensity")
            except:
                print(filepath)
                raise AssertionError


if __name__ == '__main__':
    from bacteria.code_sjh.utils.RamanData import Raman_dirwise, projectroot
    from bacteria.code_sjh.Core.basic_functions.fileReader import getRamanFromFile
    from bacteria.code_sjh.Core.basic_functions.visdom_func import *
    import os

    # def test_msc_minus_sg_norm(R, X, Au):
    #     win0 = key + "_raw"
    #     spectrum_vis(R, X, win = win0, update = None, name = key + "_exosome", vis = vis, )
    #     vis.line(np.squeeze(Au), X, win = win0, update = "append", name = key + "Au")
    #
    #     R_msc, Au_msc = msc(np.squeeze(R.numpy()), Au)
    #     # R_msc,Au_msc = numpy.squeeze(R.numpy()),np.squeeze(Au)
    #
    #     win1 = key + "_msc"
    #     l = 0.7
    #     Peaks_Au = scipy.signal.find_peaks(Au_msc,
    #                                        height = Au_msc.max() * l + Au_msc.min() * (1 - l),
    #                                        prominence = 100,
    #                                        )
    #     Peaks_R = scipy.signal.find_peaks(R_msc.mean(axis = 0),
    #                                       height = R_msc.max() * l + R_msc.min() * (1 - l),
    #                                       prominence = 100,
    #                                       )
    #     Au_msc *= Peaks_R[1]["prominences"].max() / Peaks_Au[1]["prominences"].max()
    #     spectrum_vis(R_msc, X, win = win1, name = key + "_exosome", vis = vis, )
    #     vis.line(numpy.squeeze(Au_msc), X, win = win1, update = "append", name = key + "Au", opts = dict(
    #         showlegend = True
    #     ))
    #     p = 0.5
    #     w = (R_msc > np.squeeze(Au_msc)) * p + (R_msc < np.squeeze(Au_msc)) * (1 - p)
    #     AuRemoved = (R_msc - np.squeeze(Au_msc)) * w
    #     spectrum_vis(AuRemoved, X, win = win1, update = "append", name = key + "_AuRemoved")
    #
    #     # spots = np.vstack((np.expand_dims(X[Peaks_Au[0]],axis = 0),np.expand_dims(Au_msc[Peaks_Au[0]],axis = 0)))
    #     # vis.scatter(spots.T,update = "append",name = "peaks", win = win1,                    opts = dict(
    #     #                 title = win1,
    #     #                 markersize = 5,
    #     #                 showlegend = True,
    #     #                 markersymbol = "+",
    #     #             ))
    #
    #     win2 = key + "_processed"
    #     AuProcessed = np.zeros_like(AuRemoved)
    #     Au_p = Process.preprocess_default(Au_msc)
    #     # preprocess = Process.preprocess_default
    #     preprocess = Process.process_series([
    #         # Process.bg_removal_niter_fit(num_iter = 100),
    #         Process.baseline_als(),
    #         Process.sg_filter(window_length = 17),
    #         Process.norm_func()
    #     ])
    #     for i in range(AuRemoved.shape[0]):
    #         AuProcessed[i] = preprocess(AuRemoved[i])
    #         # R_p = Process.preprocess_default(R_msc[i])
    #         # # w = (R_p > np.squeeze(Au_p)) * p + (R_p < np.squeeze(Au_p)) * (1 - p)
    #         # w = 1
    #         # AuProcessed[i] = w*R_p-Au_p
    #
    #     spectrum_vis(AuProcessed, X, win = win2, update = None, name = key + "_AuRemoved", )

    # def test_
    readdata = getRamanFromFile(wavelengthstart = 400, wavelengthend = 1800,
                                dataname2idx = {"Wavelength": 0, "Column": 1, "Intensity": 2},
                                delimeter = ",", )
    csvconfig_c = dict(dataroot = os.path.join(projectroot, "data", "liver", "liver_after_basewise", ),
                       LoadCsvFile = readdata,
                       t_v_t = [1., 0, 0],
                       backEnd = ".csv",
                       mode = "train",
                       transform = None,
                       )

    db = Raman_dirwise(**csvconfig_c)
    AuRemove(db, msc_removal)
    cfg = copy.deepcopy(csvconfig_c)
    cfg["dataroot"] = os.path.join(projectroot, "data", "liver", "liver_after_basewise")
    cfg["transform"] = Process.process_series([
        Process.interpolator(),
        Process.bg_removal_niter_fit(num_iter = 10),
        # Process.baseline_als(),
        Process.sg_filter(window_length = 11),
        Process.norm_func()
    ])

    db2 = Raman_dirwise(**cfg)
    db2.show_data_vis()
    # base2data = db.get_data_sorted_by_sample()
    # startVisdomServer()
    # vis = visdom.Visdom()
    #
    # for key in base2data.keys():
    #     if not key.startswith("s519"):
    #         continue
    #
    #     file_without_backend = key.split("-")[-1]
    #     R = base2data[key]
    #     Au, X = readdata(os.path.join(projectroot, "data", "liver", "Au", file_without_backend + ".csv"))
    #     test_msc_minus_sg_norm(R, X, Au)
    #
    # pass
