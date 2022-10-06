import glob
import copy

import numpy
import numpy as np
import pwlf
import visdom
from scipy import sparse
from scipy.sparse.linalg import spsolve

__all__ = ["baseline_als", "bg_removal_niter_fit", "bg_removal_niter_piecewisefit", "bg_removal_unfitted_area"]


# baseline_remove
def baseline_als(lam = 100000, p = 0.01, niter = 10):
    def func(y):
        L = len(y)
        t = np.diff(np.eye(L), 2)
        D = sparse.csc_matrix(t)
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
            y = y - z  # subtract the background 'z' from the original data 'y'
        return y

    return func


def bg_removal_niter_fit(num_iter = 10, degree = 4, vis = False
                         , start_end = None
                         ):
    """
	参考文章：Advances in real-time fiber-optic Raman spectroscopy
for early cancer diagnosis: Pushing the frontier into clinical
endoscopic applications
	"""
    if start_end is None:
        start_end = [400, 1800]

    def func(y):
        o = copy.deepcopy(y)
        x = numpy.linspace(*start_end, len(y))
        p = numpy.polyfit(x, y, degree)
        p = numpy.poly1d(p)
        p = p(x)
        if vis:
            viz = visdom.Visdom()
            viz.line(o, x, win = "brnf:start",
                     name = "input_tensor",
                     opts = dict(title = "brnf:start",
                                 showlegend = True,
                                 xlabel = "cm$-$1",
                                 ))
            viz.line(p, x, win = "brnf:start",
                     name = "predict",
                     update = "append",
                     opts = dict(title = "brnf:start",
                                 dash = numpy.array(['dot']),
                                 showlegend = True,
                                 xlabel = "cm$-$1",
                                 ))
        else:
            viz = vis
        for step in range(num_iter):
            o = numpy.where(o < p, o, p)
            p = numpy.polyfit(x, y, degree)
            p = numpy.poly1d(p)
            p = p(x)
            if vis:
                viz.line(o, x, win = "brnf:step-" + str(step),
                         name = "input_tensor",
                         opts = dict(title = "brnf:step-" + str(step),
                                     showlegend = True,
                                     xlabel = "cm$-$1",
                                     ))
                viz.line(p, x, win = "brnf:step-" + str(step),
                         name = "predict",
                         update = "append",
                         opts = dict(title = "brnf:step-" + str(step),
                                     dash = numpy.array(['dot']),
                                     showlegend = True,
                                     xlabel = "cm$-$1",
                                     ))
        return y - p

    return func


def bg_removal_niter_piecewisefit(num_iter = 10,
                                  degree = 4,
                                  x_seg = None,
                                  n_segments = 5,
                                  conv_cri = lambda x: False,
                                  vis = False,
                                  start_end = None):
    if start_end is None:
        start_end = [400, 1800]

    def func(y):
        if x_seg is None:
            x0 = numpy.linspace(0, 1, n_segments)
        else:
            x0 = [int(i * len(y)) for i in x_seg]

        o = copy.deepcopy(y)
        x = numpy.linspace(*start_end, len(y))
        # 多项式拟合
        my_pwlf = pwlf.PiecewiseLinFit(x, o, degree = degree)

        my_pwlf.fit_with_breaks(x0)
        p = my_pwlf.predict(x)
        res = y - p
        if vis:
            viz = visdom.Visdom()
            viz.line(o, x, win = "brnf:start",
                     name = "input_tensor",
                     opts = dict(title = "brnf:start",
                                 showlegend = True,
                                 xlabel = "cm$-$1",
                                 ytick = False,
                                 ))
            viz.line(p, x, win = "brnf:start",
                     name = "predict",
                     update = "append",
                     opts = dict(title = "brnf:start",
                                 dash = numpy.array(['dot']),
                                 showlegend = True,
                                 xlabel = "cm$-$1",
                                 ytick = False,
                                 ))
        for step in range(num_iter):
            o = numpy.where(o < p, o, p)
            my_pwlf = pwlf.PiecewiseLinFit(x, o, degree = degree)
            my_pwlf.fit_with_breaks(x0)
            p = my_pwlf.predict(x)
            res = y - p

            if vis:
                viz = visdom.Visdom()
                viz.line(o, x, win = "brnf:step-" + str(step),
                         name = "input_tensor",
                         opts = dict(title = "brnf:step-" + str(step),
                                     showlegend = True,
                                     xlabel = "cm$-$1",
                                     ytick = False,
                                     ))
                viz.line(p, x, win = "brnf:step-" + str(step),
                         name = "predict",
                         update = "append",
                         opts = dict(title = "brnf:step-" + str(step),
                                     dash = numpy.array(['dot']),
                                     showlegend = True,
                                     xlabel = "cm$-$1",
                                     ytick = False,
                                     ))
            if conv_cri(res):  # 满足准则则结束循环
                break
        return res

    return func


def bg_removal_unfitted_area():
    # TODO:将unfiited area设置为循环停止的判断句
    def func(num_iter):
        return

    return func
