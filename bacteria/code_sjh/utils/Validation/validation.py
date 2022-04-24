import numpy
import torch
import sys, os
import time
import numpy as np
from bacteria.code_sjh.utils.RamanData import *
from bacteria.code_sjh.utils.Validation.hooks import *
import visdom
from sklearn.metrics import roc_curve,auc
coderoot = os.path.split(os.path.split(__file__)[0])[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", )

try:
    from ..RamanData import Raman, RamanDatasetCore, Raman_dirwise
except:
    sys.path.append(coderoot)
    from utils.RamanData import Raman, RamanDatasetCore, Raman_dirwise


# __all__ = ["evaluate","startVisdomServer","evaluate_criteon","data2mean_std","spectrum_vis"]
def ROC(model, loader, criteon,device: torch.device,label = 0,loss_plus=None):
    """

    验证当前模型在验证集或者测试集的准确率\ROC\ TODO：分别验证不同label的准确度并可视化

    """
    model.eval()
    correct = 0
    total = len(loader.dataset)
    loss_list = []
    num_clases = loader.dataset.num_classes
    y_true_all = {}
    y_score_all = {}
    for i in range(num_clases):
        y_true_all[i]=numpy.array([])
        y_score_all[i]=numpy.array([])
    if total == 0:
        return 0
    for x, y in loader:
        x, y = x.to(device, non_blocking = True), y.to(device, non_blocking = True)
        with torch.no_grad():
            logits = model(x)   # [b,n_c]
            pred = logits.argmax(dim = 1)   #[]
        c_t = torch.eq(pred, y).sum().float().item()
        correct += c_t
        loss = criteon(output, label) if loss_plus is None else criteon(output[0], label) + loss_plus(
            *output[1:])
        loss_list.append(loss.item())
        for i in range(num_clases):
            y_true = torch.eq(y,i).detach_().int().numpy() #[b]
            y_score = logits[:,i].detach_().float().numpy() # [b]
            y_true_all[i] = np.append(y_true_all[i],y_true)
            y_score_all[i] = np.append(y_score_all[i],y_score)
    label2roc = {}
    label2auc = {}
    for i in range(num_clases):
        frp,tpr,thresholds = roc_curve(y_true,y_score)
        label2roc[i] = (frp,tpr,thresholds)
        label2auc[i] = auc(frp,tpr)

    acc = correct / total
    loss = np.mean(loss_list)
    return acc

def grad_cam(convnet,
             input,
             label = None,
             savefilepath = None, ):
    """

	:param input: [b , c = 1 ,l]
	:param label:
	:param savefilepath:
	:param win:
	:return:
	"""
    if not convnet.model_loaded:
        save_dir = savefilepath if savefilepath else os.path.join(coderoot, "checkpoints", convnet.model_name + ".mdl")
        convnet.load(save_dir)  # 加载训练模型
    input = pytorchlize(input)
    # hook
    fh = forward_hook()
    h1 = convnet.features.register_forward_hook(fh)
    bh = backward_hook()
    h2 = convnet.features.register_backward_hook(bh)

    # forward
    output = convnet(input)  # [b,n_c]

    # backward
    convnet.zero_grad()
    if label == None:
        label = torch.argmax(output)
    # class_vec
    class_loss = comp_class_vec(output, convnet.num_classes, label)
    class_loss.backward()

    fmap = fh.fmap_block[0].cpu().data.numpy().squeeze()
    grad_val = bh.grad_block[0][0].cpu().data.numpy().squeeze()  # [b,c,feature_length]

    # remove the hooks
    h1.remove()
    h2.remove()

    # cam_map
    cam = np.zeros(fmap.shape[::2], dtype = np.float)  # [b,f_l]
    ws = np.mean(grad_val, axis = (2))  # [b,c]
    bsz = ws.shape[0]
    chs = ws.shape[1]
    for b in range(bsz):
        for c in range(chs):
            w = ws[b, c]
            cam[b] += w * fmap[b, c, :]  # [b] * [b,l]
            cam[b] = np.where(cam[b] > 0, cam[b], 0)
            cam[b] -= cam[b].min()
            cam[b] /= cam[b].max()

    return cam  # [b,l]

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


def evaluate_loss(model, loader, criteon, device,loss_plus: callable = None):
    """
    验证当前模型在验证集或者测试集的准确率
    """
    loss_list = []
    for x, y in loader:
        x, y = x.to(device, non_blocking = True), y.to(device, non_blocking = True)
        with torch.no_grad():
            output = model(x)
            loss = criteon(output, label) if loss_plus is None else criteon(output[0], label) + loss_plus(
                *output[1:])
            loss_list.append(loss.item())
    return np.mean(loss_list)


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
        return out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds = roc_curve(y, scores, )
    plt.plot(fpr,tpr)
    plt.show()