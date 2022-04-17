import torch
import sys, os
import time
import numpy as np
import visdom

coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", )

try:
    from ..RamanData import Raman, RamanDatasetCore, Raman_dirwise
except:
    sys.path.append(coderoot)
    from utils.RamanData import Raman, RamanDatasetCore, Raman_dirwise


# __all__ = ["evaluate","startVisdomServer","evaluate_criteon","data2mean_std","spectrum_vis"]

def evaluate(model, loader, device: torch.device):
    """

    验证当前模型在验证集或者测试集的准确率 TODO：分别验证不同label的准确度并可视化

    """
    model.eval()
    correct = 0
    total = len(loader.dataset)
    if total == 0:
        return 0
    for x, y in loader:
        x, y = x.to(device, non_blocking = True), y.to(device, non_blocking = True)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim = 1)
        c_t = torch.eq(pred, y).sum().float().item()
        correct += c_t
    acc = correct / total
    return acc


def evaluate_labelwise(model, dataset: RamanDatasetCore, device: torch.device, viz: visdom.Visdom = None):
    model.eval()
    label2data = dataset.get_data_sorted_by_label()
    label2name = dataset.label2name()
    label2acc = {}
    for k in label2data.keys():
        data = label2data[k]
        name = label2name[k]
        labels = torch.ones(data.shape[0]) * k

        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(data)
            pred = logits.argmax(dim = 1)
        c_t = torch.eq(pred, labels).sum().float().item()
        label2acc[name] = c_t / pred.shape[0]
    return label2acc


def evaluate_samplewise(model, dataset: Raman_dirwise or dict, device: torch.device):
    # TODO:
    t0 = time.time()
    model.eval()
    sample2acc = {}
    sample2label = dataset.sample2label()
    sample2data = dataset.get_data_sorted_by_sample()
    t1 = time.time()
    if t1 - t0 > 1:
        print("get_data_time:", t1 - t0)
    for k in sample2data.keys():
        t0 = time.time()
        data = sample2data[k]
        labels = torch.ones(data.shape[0]) * sample2label[k]
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(data)
            pred = logits.argmax(dim = 1)
        c_t = torch.eq(pred, labels).sum().float().item()
        sample2acc[k] = c_t / pred.shape[0]
        t1 = time.time()
        if t1 - t0 > 1:
            print("cal_acc_time_{}:".format(k), t1 - t0)
    return sample2acc


def evaluate_criteon(model, loader, criteon, device):
    """
    验证当前模型在验证集或者测试集的准确率
    """
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device, non_blocking = True), y.to(device, non_blocking = True)
        with torch.no_grad():
            logits = model(x)
            correct += criteon(logits, y)
    acc = correct / total
    return acc


def comp_class_vec(output, num_classes, index = None, ):
    """

    :param output:[b,n_c]
    :param index: [b] or int
    :return: class_vec
    """
    batchsize = output.shape[0]
    if index is None:
        index = torch.argmax(output)  # [b]
    elif type(index) is int:
        index = torch.ones(batchsize) * index

    index = torch.unsqueeze(index, 1).to(torch.int64)
    one_hot = torch.zeros(batchsize, num_classes).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.mean(one_hot * output)

    return class_vec


class encode():
    def __init__(self, module, pthfile):
        self.module = module
        self.module.load(pthfile)
        self.module.eval()

    def __call__(self, input):
        out = self.module(input)
        out.detach_()


if __name__ == '__main__':
    output = torch.randn(16, 6)  # [b,num_c]
    label = torch.ones(16).to(torch.int64)  # [b]
    label[3] = 3
    v = comp_class_vec(output, 6, label)
