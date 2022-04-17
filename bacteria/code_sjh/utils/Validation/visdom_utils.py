import copy
import time
import numpy
import torch
import sys, os
import requests, webbrowser
import visdom

coderoot = os.path.dirname(__file__)
for i in range(2):
    coderoot = os.path.dirname(coderoot)

projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", )
sys.path.append(coderoot)


def startVisdomServer(url = "http://localhost:8097"):
    try:
        response = requests.get(url, timeout = 5).status_code
        if response != 200:

            os.system("start python -m visdom.server")  # 启动服务器
            time.sleep(10)
            print("server started!")
            webbrowser.open(url)
        else:
            print("server aready started,url = ", url)
    except:
        os.system("start python -m visdom.server")  # 启动服务器
        time.sleep(3)
        print("server started!")
        webbrowser.open(url)
    finally:
        return


def data2mean_std(spectrums: torch.Tensor or numpy.ndarray):
    """
    :param spectrums:a batch of spectrums[b,l] or [b,1,l]
    :returns spectrum_mean ,spectrum_up=spectrum_mean+std,spectrum_down=spectrum_mean-std
    """
    if type(spectrums) == numpy.ndarray:
        spectrums = torch.tensor(spectrums)
    if len(spectrums.shape) == 3 and spectrums.shape[1] == 1:
        spectrums = torch.squeeze(spectrums, dim = 1)
    elif len(spectrums.shape) > 2:
        raise AssertionError

    lenth = spectrums.shape[-1]
    batchsz = spectrums.shape[0]

    y_mean = torch.Tensor(lenth)
    y_up = torch.Tensor(lenth)
    y_down = torch.Tensor(lenth)
    for idx in range(lenth):
        p = spectrums[:, idx]
        mean = p.mean()
        std = p.std()
        y_mean[idx] = mean
        y_up[idx] = mean + std
        y_down[idx] = mean - std
    return y_mean, y_up, y_down


def batch_plt(name2value: dict,
              x,
              win = 'batch_plt_vis_default',
              update = None,
              viz: visdom.Visdom = None,
              opts: dict = None,
              name2opts: dict = None):
    # assert all([type(k) == str for k in name2value.keys()]), "all names must be str,but got {}".format(
    #     [type(k) for k in name2value.keys()])
    # assert all([type(k) == (float or torch.Tensor or numpy.ndarray) for k in
    #             name2value.values()]), "all values must be float or Tensor,but got {}".format(
    #     [type(k) for k in name2value.values()])
    if viz == None:
        viz = visdom.Visdom()
    if opts is None:
        opts = dict(title = win, showlegend = True)
    for k in sorted(name2value.keys()):

        viz.line(Y = [name2value[k]], X = [x], win = win, update = update, name = str(k) if type(k) !=str else k,
                 opts = opts if name2opts is None else name2opts[k])
    return


def spectrum_vis(spectrums: torch.Tensor,
                 xs = None,
                 win = 'spectrum_vis_win_default',
                 update = None,
                 name = None,
                 vis: visdom.Visdom = None,
                 opts: dict = None,
                 bias = 0,
                 side = True):
    """

    :param spectrums: a batch of spectrums[b,l] or [b,1,l]
    :param xs: the x axis of the spectrum, default is None
    :param win: the window name in visdom localhost and the title of the window
    :param update: None if to rewrite and "append" if to add
    :param name: the name of the batch of spectrums, the line of the mean value is named as [name]+"_mean", mean + std
    -> [name] + "_up" , mean - std -> [name] + "_down"
    for example : If the name of the spectrums is "bacteria_R",then the names of the lines are "bacteria_R_mean",
    "bacteria_R_up" and "bacteria_R_down"
    :param vis: visdom.Visdom, a new one will be created if None
    :param opts: dict, {"main":dict(line visdom opts for the "mean" line), "side":dict(line visdom opts for the "up" line
     and "down" line)}
    :param bias: move the lines of spectrums up(positive number) or down(negative number)
    :param side: whether to show the mean + std and mean - std line
    :returns spectrum_mean ,spectrum_up=spectrum_mean+std,spectrum_down=spectrum_mean-std
    """
    if name == None:
        name = ""
    if opts is None:
        opts = dict(main = dict(
            xlabel = "cm-1",
        ),
            side = dict(
                xlabel = "cm-1",
                dash = numpy.array(['dot']),
                linecolor = numpy.array([[123, 104, 238]]),
            ))
    if not "main" in opts.keys() or not "side" in opts.keys():
        opts = dict(
            main = opts,
            side = opts
        )
    opts["main"].update(dict(title = win))
    opts["side"].update(dict(title = win))
    opts_down = copy.deepcopy(opts["side"])

    if vis is None:
        vis = visdom.Visdom()
    y_mean, y_up, y_down = data2mean_std(spectrums)
    if xs is None:
        xs = torch.arange(spectrums.shape[-1])

    assert xs.shape[-1] == y_mean.shape[-1], r"lenth of xs and spectrums doesn't fit"
    vis.line(X = xs, Y = y_mean + bias,
             win = win,
             name = name + "_mean",
             update = update,

             opts = opts["main"]
             )
    if side:
        vis.line(X = xs, Y = y_up + bias,
                 win = win,
                 update = "append",
                 name = name + "_up",
                 opts = opts["side"], )

        vis.line(X = xs, Y = y_down + bias,
                 win = win,
                 update = "append",
                 name = name + "_down",
                 opts = opts_down, )

        return y_mean, y_up, y_down


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from utils import LamenData

    startVisdomServer()
    tvt = [0.8, 0.2, 0]

    dataname = "data_AST"

    train_db = LamenData.Raman(os.path.join(dataroot, dataname), mode = "train", backEnd = "-.csv", t_v_t = tvt)
    train_loader = DataLoader(train_db, batch_size = 16, shuffle = True, num_workers = 0)
    spectrum = iter(train_loader).__next__()[0]
    spectrum_vis(spectrum)
