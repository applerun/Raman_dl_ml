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

from models.BasicModule import BasicModule, Flat

# 可视化
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

__all__ = []



class AlexDense(BasicModule):
    '''
    只由一层全连接层分类

    '''

    def __init__(self, tensor_sample, num_classes = 6, dropout = 0.1):
        """
        需要输入一个样例数据以生成网络 用作样例的数据仍然可以用作训练或验证

        """

        super(AlexDense, self).__init__()
        self.line_num = 0
        self.model_name = 'dense_net'
        self.dropout = dropout
        self.num_classes = num_classes

        self.flat = Flat()

        tensor_sample2 = tensor_sample.clone()
        tensor_sample2 = self.flat(tensor_sample2)
        self.buid_classifier(tensor_sample2)

    def buid_classifier(self, x: torch.Tensor):
        length = x.shape[-1]
        self.classifier = nn.Sequential(

            nn.Linear(length, 64),
            nn.ReLU(inplace = True),
            nn.Dropout(self.dropout),

            nn.Linear(64, 32),
            nn.ReLU(inplace = True),
            nn.Dropout(self.dropout),

            nn.Linear(32, self.num_classes),

        )

    def forward(self, x):
        x = self.flat(x)
        x = self.classifier(x)
        return x

    def weight_vis(self, savefilepath = None):
        """
        Param shapefile: 训练后模型的存储路径，若无则从checkpoint中寻找
        """
        save_dir = savefilepath if savefilepath else os.path.join(coderoot, "checkpoints", self.model_name + ".mdl")
        self.load(save_dir)
        vis = visdom.Visdom()
        for name, parameters in self.classifier.named_parameters():
            print(name, ":", parameters.shape)
        pass

    def dense_vis(self, savefilepath = None):
        if not self.model_loaded:
            save_dir = savefilepath if savefilepath else os.path.join(coderoot, "checkpoints", self.model_name + ".mdl")
            self.load(save_dir)  # 加载训练模型

        channels = 1
        para = None
        for name, parameter in self.classifier.named_parameters():
            if not name.endswith("weight"):
                continue
            if para != None:
                para = parameter @ para
            else:
                para = parameter
        # para：[2,lenth]

        lenth = para.shape[-1]

        out = torch.zeros(lenth)  # [length]
        diff = torch.Tensor(lenth)  # 计算差异性：MSE
        for l in range(lenth):
            diff[l] = para[:, l].std() * 40

        xs = torch.linspace(400, 1800, lenth)
        for l in range(channels):
            ys = diff[l * lenth:(l + 1) * lenth]
            # ys -= torch.tensor(5)
            vis.line(X = xs, Y = ys + l * 5,
                     win = "dense" + str(l),
                     update = "append",
                     # env = "main_dense",
                     name = "num_channel ",
                     opts = dict(title = "dense" + str(l)))
        self.line_num += 1

        # 画出所有num_classes的dense
        for i in range(self.num_classes):
            # 计算
            for l in range(lenth):
                out[l] = para[i, l] * 40 - torch.tensor((i + 1) * 5)
            # 绘图
            for l in range(channels):
                ys = out[l * lenth:(l + 1) * lenth]
                # ys -= torch.tensor(5)
                vis.line(X = xs, Y = ys,
                         win = "channel:" + str(l),
                         # env = "main_filteredSignal",
                         update = "append",
                         name = "num" + str(self.line_num),
                         opts = dict(title = "channel:" + str(l)))
            self.line_num += 1
        return

    def pred_label(self, sample_tensor):  # [1,l]
        # 获取预测值
        with torch.no_grad():
            logits = self.forward(torch.unsqueeze(sample_tensor, dim = 0))  # [b,6]
            pred = logits.argmax(dim = 1)  # [b]
        return pred


if __name__ == '__main__':
    from utils.LamenDataLoader import Lamen
    from utils.validation import startVisdomServer

    startVisdomServer()
    test_db = Lamen(dataroot, mode = "test")

    # Sample
    sample_tensor, sample_label = test_db.__getitem__(1)
    vis = visdom.Visdom()
    vis.line(torch.squeeze(sample_tensor), win = "sample_x",
             opts = dict(title = "sample_x:label=" + str(sample_label.item())))

    sample_tensor = torch.unsqueeze(sample_tensor, dim = 0)  # [b=1,c=1,dim=length]
    alexnet = AlexDense(tensor_sample = sample_tensor, num_classes = 2)
    model_name = "alexnet_AST"
    alexnet.set_model_name(model_name)

    save_dir = os.path.join(coderoot, "checkpoints", alexnet.model_name + ".mdl")
    alexnet.eval()
    alexnet.load(save_dir)
    alexnet.weight_vis()
    alexnet.dense_vis()
    instances = len(test_db)
    for idx in range(instances):
        sample_tensor, sample_label = test_db.__getitem__(idx)
        vis.line(
            X = torch.linspace(400, 1800, sample_tensor.shape[-1]),
            Y = torch.squeeze(sample_tensor) + sample_label, win = "sampleinstances", update = "append",
            opts = dict(title = "samples"), name = str(idx))
# alexnet.feature_vis(sample_label, sample_tensor)
# print("Process:{}/{}".format(idx, instances))
