import numpy
import torch
import visdom
from torch import nn, optim

# 记录各个路径
import os, sys

coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "data_AST")
sys.path.append(coderoot)

from utils.LamenDataLoader import Lamen
from utils.validation import startVisdomServer
from utils import Process

test_db = Lamen(dataroot, mode = "test",transform=None)
startVisdomServer()
sample_tensor, sample_label = test_db.__getitem__(1)
sample_tensor = torch.squeeze(sample_tensor)
sample_tensor = sample_tensor.detach().numpy()
vis = visdom.Visdom()
x = numpy.linspace(400,1800,len(sample_tensor))
vis.line(sample_tensor,x,win = "unprocessesd",opts = {"title":"unprocessed","showlegend":True})

sgf = Process.sg_filter()

nf = Process.norm_func()

bals = Process.baseline_als() #孙学长的预处理函数

brmi = Process.bg_removal_niter_fit(vis = True) #黄志伟老师的预处理函数？
brnp = Process.bg_removal_niter_piecewisefit(vis = True) #使用分段多项式拟合
# sample_tensor = sgf(nf(sample_tensor))

y = bals(sample_tensor)
x = numpy.linspace(400,1800,len(y))
print(len(y))
vis.line(X = x,Y = nf(sgf(y)),win = "Sun_processeed",
                    opts = dict(
                    title = "Sun_processeed",
                    ))
vis.line(X = x,Y=sample_tensor-y,win = "unprocessesd",
                    update = "append",
                    name = "bals_baseline",
                    opts = dict(
                    showlegend = True,
                    ))

y1 = brmi(sample_tensor)
x = numpy.linspace(400,1800,len(y1))
print(len(y1))
vis.line(nf(sgf(y1)),x,win = "huang_processed",
         opts = {"title":"huang_processed"})

vis.line(sample_tensor-y1,x,win = "unprocessesd",
                    update = "append",
                    name = "brnf_baseline",
                    opts = dict(
                    showlegend = True,
                    ))

y2 = brnp(sample_tensor)
x = numpy.linspace(400,1800,len(y2))
print(len(y2))
vis.line(nf(sgf(y)),x,win = "piecewise_processed",
         opts = {"title":"picewise_processed"})
vis.line(sample_tensor-y2,x,win = "unprocessesd",
                    update = "append",
                    name = "brnp_baseline",
                    opts = dict(
                    showlegend = True,
                    ))


brnp = Process.bg_removal_niter_piecewisefit(num_iter = 3) #使用分段多项式拟合
y2 = brnp(sample_tensor)
x = numpy.linspace(400,1800,len(y2))
print(len(y2))
vis.line(nf(sgf(y)),x,win = "piecewise_processed_n-5",
         opts = {"title":"picewise_processed_n-5"})
vis.line(sample_tensor-y2,x,win = "unprocessesd",
                    update = "append",
                    name = "brnp_baseline_n-5",
                    opts = dict(
                    showlegend = True,
                    ))

brnp = Process.bg_removal_niter_piecewisefit(n_segments = 10) #使用分段多项式拟合
y2 = brnp(sample_tensor)
x = numpy.linspace(400,1800,len(y2))
print(len(y2))
vis.line(nf(sgf(y)),x,win = "piecewise_processed_s-5",
         opts = {"title":"picewise_processed_s-5"})
vis.line(sample_tensor-y2,x,win = "unprocessesd",
                    update = "append",
                    name = "brnp_baseline_s-5",
                    opts = dict(
                    showlegend = True,
                    ))
