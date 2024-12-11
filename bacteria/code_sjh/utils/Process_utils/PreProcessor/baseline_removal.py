import numpy as np
import pwlf
from scipy import sparse
from scipy.sparse.linalg import spsolve
from bacteria.code_sjh.utils.Process_utils.Core import *

__all__ = ["baseline_als", "bg_removal_niter_fit", "bg_removal_niter_piecewisefit", "bg_removal_unfitted_area","airALS"]


# baseline_remove
class baseline_als(ProcessorFunction):
    def __init__(self, lam = 100000, p = 0.01, niter = 10):
        """
        非对称最小二乘法
        参考文献：https://eigenvector.com/wp-content/uploads/2020/01/WhittakerSmoother.pdf
        @param lam: \lambda
        @param p: 惩罚系数
        @param niter: 迭代次数
        """
        super(baseline_als, self).__init__("baseline_als(lam={},p={}，niter={}".format(lam, p, niter))
        self.p = p
        self.niter = niter
        self.lam = lam

    def __call__(self, y, x = None):
        L = len(y)
        t = np.diff(np.eye(L), 2)
        D = sparse.csc_matrix(t)
        w = np.ones(L)
        for i in range(self.niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + self.lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = self.p * (y > z) + (1 - self.p) * (y < z)
            y = y - z  # subtract the background 'z' from the original data 'y'
        return y if x is None else (y, x)


class airALS(ProcessorFunction):
    def __init__(self, lam = 1e5, p = 0.01, niter = 10):
        """
        一种给改进的非对称最小二乘法（个人感觉提升不大）
        参考文献：https://doi.org/10.1038/srep39891
        @param lam: \lamdba
        @param p: 惩罚系数
        @param niter: 迭代次数
        """
        super(airALS, self).__init__("airALS(lam={},p={},niter={})".format(lam, p, niter))
        self.lam = lam
        self.p = p
        self.niter = niter

    def __call__(self, y, x = None):
        L = len(y)
        t = np.diff(np.eye(L), 2)
        D = sparse.csc_matrix(t)
        w = np.ones(L)

        for i in range(self.niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + self.lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)

            d = y - z
            dt = abs(sum(d * (d < 0)))
            if dt < sum(0.001 * abs(y)):
                break
            w = self.p * (y > z) + (1 - self.p) * (y < z)
            w = w * i * numpy.exp(numpy.abs(d) / dt) # 相比BALS添加了这一步运算

            y = d  # subtract the background 'z' from the original data 'y'
        return y if x is None else (y, x)


class bg_removal_niter_fit(ProcessorFunction):
    def __init__(self, niter = 10, degree = 4, vis = False
                 ):

        """
        多项式迭代拟合法（终止条件为循环次数）
        参考文章：Advances in real-time fiber-optic Raman spectroscopy
    for early cancer diagnosis: Pushing the frontier into clinical
    endoscopic applications
        链接：https://onlinelibrary.wiley.com/doi/full/10.1002/tbio.202000018
        """
        super(bg_removal_niter_fit, self).__init__(
            "bg_removal_niter_fit(niter={},degree={})".format(niter, degree))

        self.degree = degree
        self.vis = vis
        self.niter = niter

    def __call__(self, y, x = None):
        o = copy.deepcopy(y)
        x_ = numpy.linspace(0, 1, len(y)) if x is None else x
        p = numpy.polyfit(x_, y, self.degree)
        p = numpy.poly1d(p)
        p = p(x_)

        for step in range(self.niter):
            o = numpy.where(o < p, o, p)
            p = numpy.polyfit(x_, y, self.degree)
            p = numpy.poly1d(p)
            p = p(x_)

        if x is None:
            return y - p
        else:
            return y - p, x


class bg_removal_niter_piecewisefit(ProcessorFunction):
    def __init__(self, num_iter = 100, degree = 4, x_seg = None, n_segments = 5,
                 ):
        """
        分段多项式迭代拟合法
        @param num_iter: 迭代次数
        @param degree: 拟合阶数
        @param x_seg: 分段横坐标（每段边界位置）
        @param n_segments:  分段数量（默认每段长度一样）
        """

        self.num_iter = num_iter
        self.degree = degree
        self.x_seg = x_seg
        self.n_segments = n_segments
        super(bg_removal_niter_piecewisefit, self).__init__(
            "bg_removal_niter_piecewisefit(num_iter = {},degree = {},".format(self.num_iter, self.degree)
            + ("n_segments = {})".format(self.n_segments) if self.x_seg is None
               else "x_seg ={})".format(self.x_seg)))

    def __call__(self, y, x = None):
        o = copy.deepcopy(y)
        x_ = numpy.linspace(0, 1, len(y)) if x is None else x
        if self.x_seg is None:
            self.x_seg = numpy.linspace(min(x_), max(x_), self.n_segments)

        # 多项式拟合
        my_pwlf = pwlf.PiecewiseLinFit(x_, o, degree = self.degree)
        my_pwlf.fit_with_breaks(self.x_seg)
        p = my_pwlf.predict(x_)
        res = y - p

        for step in range(self.num_iter):
            o = numpy.where(o < p, o, p)
            my_pwlf = pwlf.PiecewiseLinFit(x_, o, degree = self.degree)
            my_pwlf.fit_with_breaks(self.x_seg)
            p = my_pwlf.predict(x_)
            res = y - p
        return res if x is None else (res, x)





if __name__ == '__main__':
    a = bg_removal_niter_piecewisefit(11, 3, x_seg = [0, 1, 2], n_segments = 4)
    b = eval(str(a))
    print(b)
    pass
