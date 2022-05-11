import numpy
import torch
import visdom

# 记录各个路径
import os, sys

coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "liver", "liver_all_samplewise")
sys.path.append(coderoot)

from bacteria.code_sjh.utils.RamanData import Raman, Raman_dirwise
from bacteria.code_sjh.utils.Validation.visdom_utils import startVisdomServer
from bacteria.code_sjh.utils import Process

test_db = Raman_dirwise(dataroot, mode = "train", transform = None)
startVisdomServer()
sample_tensor, sample_label = test_db.__getitem__(1)
sample_tensor = torch.squeeze(sample_tensor)
sample_tensor = sample_tensor.detach().numpy()
vis = visdom.Visdom()

sgf = Process.sg_filter()

nf = Process.norm_func()

bals = Process.baseline_als()  # 孙学长的预处理函数

brmi = Process.bg_removal_niter_fit(vis = False)  # 黄志伟老师的预处理函数？
brnp = Process.bg_removal_niter_piecewisefit(vis = False)  # 使用分段多项式拟合
# sample_tensor = sgf(nf(sample_tensor))

y = bals(sample_tensor)
x = test_db.xs
print(len(y))
vis.line(X = x, Y = y, win = "processed", name = "非对称最小二乘法拟合",
         opts = dict(
	         title = "去除基线效果",
	         showlegend = True,
	         xlabel = "Wavenumber cm-1",
	         ylabel = "intensity"
         ))
vis.line(X = x, Y = sample_tensor - y, win = "unprocessesd",
         name = "非对称最小二乘法拟合", opts = dict(title = "基线拟合效果", showlegend = True,
                                          xlabel = "Wavenumber cm-1",
                                          ylabel = "intensity"
                                          ),
         )

y1 = brmi(sample_tensor)

print(len(y1))
vis.line(y1, x, win = "processed", name = "多项式迭代拟合", update = "append",
         )

vis.line(sample_tensor - y1, x, win = "unprocessesd",
         update = "append",
         name = "多项式迭代拟合",
         )

y2 = brnp(sample_tensor)

print(len(y2))
vis.line(y2, x, win = "processed", name = "分段多项式迭代拟合", update = "append",
         )
vis.line(sample_tensor - y2, x, win = "unprocessesd",
         update = "append",
         name = "分段多项式迭代拟合",
         )

vis.line(sample_tensor, x, win = "unprocessesd", name = "原始数据", update = "append",
         opts = dict(linecolor = numpy.array([[0, 0, 0]])))

vis.line(y, x, win = "sgfilter", name = "s-g滤波前", opts = dict(title = "s-g平滑效果", showlegend = True, xlabel = "Wavenumber cm-1",
             ylabel = "intensity"))
vis.line(sgf(y), x, win = "sgfilter", name = "s-g滤波后", update = "append")

# brnf = Process.bg_removal_niter_piecewisefit(num_iter = 3)  # 使用分段多项式拟合
# y2 = brnf(sample_tensor)
# x = numpy.linspace(400, 1800, len(y2))
# print(len(y2))
# vis.line(nf(sgf(y)), x, win = "piecewise_processed_n-5",
#          opts = {"title": "picewise_processed_n-5"})
# vis.line(sample_tensor - y2, x, win = "unprocessesd",
#          update = "append",
#          name = "brnp_baseline_n-5",
#          )
#
# brnf = Process.bg_removal_niter_piecewisefit(n_segments = 10)  # 使用分段多项式拟合
# y2 = brnf(sample_tensor)
# x = numpy.linspace(400, 1800, len(y2))
# print(len(y2))
# vis.line(nf(sgf(y)), x, win = "piecewise_processed_s-5",
#          opts = {"title": "picewise_processed_s-5"})
# vis.line(sample_tensor - y2, x, win = "unprocessesd",
#          update = "append",
#          name = "brnp_baseline_s-5",
#          opts = dict(
# 	         showlegend = True,
#          ))
