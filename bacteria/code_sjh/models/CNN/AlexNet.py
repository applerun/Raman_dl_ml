import numpy
import torch
import visdom
from torch import nn

# 记录各个路径
import os

coderoot = os.path.split(os.path.split(__file__)[0])[0]
coderoot = os.path.split(coderoot)[0]
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "data_AST")

from bacteria.code_sjh.models.BasicModule import BasicModule, Flat
from bacteria.code_sjh.utils.RamanData import pytorchlize

__all__ = ["AlexNet_Sun"]


def AlexNet_Sun_generator(dropout = 0.1, dense_bias = True, alpha = 0.15):
    class AlexNet(BasicModule):
        '''
        参考《基于拉曼光谱技术和深度学习算法的细菌鉴定》-孙宏奕
        '''

        def __init__(self, sample_tensor: torch.Tensor, num_classes):
            """
            需要输入一个样例数据以生成网络 用作样例的数据仍然可以用作训练或验证
            :param sample_tensor:样例数据[1,]

            """
            self.alpha = alpha
            super(AlexNet, self).__init__(sample_tensor)
            sample_tensor = pytorchlize(sample_tensor)
            self.model_name = 'alexnet'
            self.dropout = dropout
            self.num_classes = num_classes

            self.features = nn.Sequential(
                nn.Conv1d(sample_tensor.shape[-2], 8, kernel_size = tuple([8]), padding = 0),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(alpha, inplace = True),
                nn.MaxPool1d(kernel_size = 2, stride = 2),

                nn.Conv1d(8, 16, kernel_size = tuple([16]), padding = 0),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(alpha, inplace = True),
                nn.MaxPool1d(kernel_size = 2, stride = 2),
            )
            self.flat = Flat()
            tensor_sample2 = sample_tensor.clone()
            tensor_sample2 = self.flat(self.features(tensor_sample2))
            self.buid_classifier(tensor_sample2, bias = dense_bias)

        def buid_classifier(self, x: torch.Tensor, bias = True):
            length = x.shape[-1]
            self.classifier = nn.Sequential(

                nn.Linear(length, 64, bias = bias),
                nn.LeakyReLU(self.alpha, inplace = True),
                nn.Dropout(self.dropout),

                nn.Linear(64, 32, bias = bias),
                nn.LeakyReLU(self.alpha, inplace = True),
                nn.Dropout(self.dropout),

                nn.Linear(32, self.num_classes, bias = bias),
                # nn.Sigmoid()
                # nn.Softmax()
            )

        def forward(self, x):
            x = self.features(x)
            x = self.flat(x)
            x = self.classifier(x)
            return x

        # ———— visualizations———— #
        def feature_vis(self, sample_label, sample_tensor: torch.Tensor, savefilepath = None, win = "feature_vis",
                        update = None, pass_wrong = True, vis = None, xs = None):
            """

            :param sample_label: 需要提取特征的数据的label
            :param sample_tensor: 需要提取特征的数据
            :param savefilepath: path of .mdl(or.pth) file
            :param win: the window name in visdom localhost and the title of the window
            :param update: None if to rewrite and "append" if to add
            :param vis: visdom.Visdom, a new one will be created if None
            :param pass_wrong: ignore the spectrum if the model can't classify it correctly
            :param xs:
            :return:

                input: 一个信号数据
                output:
                    画出卷积层输出的16个channel的信号：
                    在每个channel对应的visdom窗口（win=channel:{channel_idx}）画出信号(+label*5)的图形
            """
            if vis == None:
                vis = visdom.Visdom()
            if not self.model_loaded:
                save_dir = savefilepath if savefilepath else os.path.join(coderoot, "checkpoints",
                                                                          self.model_name + ".mdl")
                self.load(save_dir)  # 加载训练模型
            batchsz = 1
            # 将样例数据encode

            if len(sample_tensor.shape) == 2:
                sample_tensor = torch.unsqueeze(sample_tensor, dim = 0)  # [b=1,c=1,dim=length]
            elif len(sample_tensor.shape) == 3:
                batchsz = sample_tensor.shape[0]
            if type(sample_label) == int:
                sample_label = sample_label * torch.ones(batchsz)
            feature = self.features(sample_tensor)  # [b,c=16,lenth]

            # 待可视化的decode前的数据
            sample_out = feature  # [c=16,dim=length]

            # 将数据decode，并作出预测
            if pass_wrong:
                label_out = self.flat(feature)  # [b,c*lenth]
                label_out = self.classifier(label_out)  # [b,num_classes]
                label_out = label_out.argmax(dim = 1).detach()  # [b]
            else:  # 如果不需要跳过错误预测的数据则省略该步骤
                label_out = sample_label

            # 提取通道数、信号长度、信号、标签
            channels = sample_out.shape[-2]
            lenth = sample_out.shape[-1]
            label = sample_label.detach()  # [b]

            # 设置图形

            vis = visdom.Visdom()

            # 横坐标
            if xs is None:
                xs = torch.linspace(400, 1800, lenth)

            # 每个label 分别画图
            spectrum_each_label = {}
            for i in range(self.num_classes):
                spectrum_each_label[i] = None
            # 分离每个label的信号
            for i in range(batchsz):
                if label[i] != label_out[i]:  # 仅取预测正确的信号
                    continue
                if spectrum_each_label[label[i].item()] is None:
                    spectrum_each_label[label[i].item()] = torch.unsqueeze(sample_out[i, :, :],
                                                                           dim = 0)
                else:
                    spectrum_each_label[label[i].item()] = torch.cat(
                        (spectrum_each_label[label[i].item()],
                         torch.unsqueeze(sample_out[i, :, :],
                                         dim = 0)),
                        dim = 0
                    )
            # spectrum_each_label {labels:[tensor-[b,c,d]]}
            # 每个channel每个label一条线
            for channel_idx in range(channels):
                for label_num in range(self.num_classes):
                    bias = torch.tensor(label_num * 5 + 5)
                    spectrums = spectrum_each_label[label_num][:, channel_idx,
                                :]  # [num_spctrm_of_l,c,d] -> [num_spctrm_of_l,1,d]
                    opts = {}
                    opts["main"] = dict(ytick = False,
                                        showlegend = True,
                                        ytickmin = 0,
                                        ytickmax = 5 * (self.num_classes + 1),
                                        ytickstep = 5,
                                        yticklabels = ["diff"] + ["label" + str(l) for l in range(self.num_classes)],
                                        xlabel = "cm-1",
                                        )
                    opts["side"] = dict(
                        dash = numpy.array(['dot']),
                        linecolor = numpy.array([[123, 104, 238]]),
                        ytick = True,
                        ytickmin = 0,
                        ytickmax = 5 * self.num_classes,
                        ytickstep = 5,
                        yticklabels = ["diff"] + ["label" + str(l) for l in range(self.num_classes)],
                        xlabel = "cm-1",
                    )
                    visdom_func.spectrum_vis(
                        spectrums,
                        xs,
                        win + "_channel_" + str(channel_idx),
                        update = update if label_num == 0 else "append",
                        name = "label-" + str(label_num),
                        vis = vis,
                        bias = bias,
                        opts = opts
                    )
            return spectrum_each_label

        def dense_vis(self, savefilepath = None, win = "feature_vis", update = None, vis = None, xs = None):
            """

            :param savefilepath: path of .mdl(or.pth) file
            :param win: the window name in visdom localhost and the title of the window
            :param update: None if to rewrite and "append" if to add
            :param vis: visdom.Visdom, a new one will be created if None
            :param xs: the x axis of the spectrum, default is None
            :return: diff(std,dim=1) of the weight@weight@weight in the dense classifier

            vis图像：
            忽略全连接层的bias，仅根据weight画出
            weight权值：weight1@weight2@... = tensor(shape=[分类数量，信号长度])
            output1：在每个channel对应的visdom窗口（win=channel:{channel_idx}）画出weight权值(-label*5)的图形
            output2:在每个channel对应的visdom窗口（win=channel:{channel_idx}）画出weight权值的差异统计量(标准差——越大证明该处信号在分类中的作用越大)

            """
            if vis == None:
                vis = visdom.Visdom()
            if not self.model_loaded:
                save_dir = savefilepath if savefilepath is not None else os.path.join(coderoot, "checkpoints",
                                                                                      self.model_name + ".mdl")
                self.load(save_dir)  # 加载训练模型
            channels = 16
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
            single_lenth = lenth // 16

            diff = torch.Tensor(lenth)  # 计算差异性：MSE
            for l in range(lenth):
                diff[l] = para[:, l].std()

            diff *= 5 / max(diff)  # 标准化
            if xs is None:
                xs = torch.linspace(400, 1800, single_lenth)
            for l in range(channels):
                ys = diff[l * single_lenth:(l + 1) * single_lenth]

                vis.line(X = xs, Y = ys,
                         win = win + "_channel_" + str(l),
                         update = update,

                         name = "diff",
                         opts = dict(title = win + "_channel_" + str(l),
                                     ytick = True,
                                     ytickmin = 0,
                                     ytickmax = 5 * channels // 4,
                                     ytickstep = 5,
                                     yticklabels = ["diff"] + ["label" + str(l) for l in range(self.num_classes)],
                                     xlabel = "cm-1", ),
                         )
                vis.line(X = xs, Y = ys + (l % 4) * 5,
                         win = win + "_dense" + str(l // 4),
                         update = None if l % 4 == 0 else "append",

                         name = "num_channel " + str(l),
                         opts = dict(title = win + "_dense" + str(l // 4),
                                     ytick = True,
                                     ytickmin = 0,
                                     ytickmax = 5 * channels // 4,
                                     ytickstep = 5,
                                     yticklabels = ["diff"] + ["label" + str(l) for l in range(self.num_classes)],
                                     xlabel = "cm-1", )
                         )

            return diff

        def pred_label(self, sample_tensor):  # [1,l]
            """
            获取预测值
            :param sample_tensor: 需要预测的数据
            :return: 预测的结果
            """

            # 将sampletensor
            ls = len(sample_tensor.shape)
            if ls > 4:
                sample_tensor = torch.squeeze(sample_tensor)
            assert ls < 4, "too many dimensions!"
            for i in range(3 - ls):
                sample_tensor = torch.unsqueeze(sample_tensor, dim = ls - 1)

            with torch.no_grad():
                logits = self.forward(sample_tensor)  # [b,num_classes]
                pred = logits.argmax(dim = 1)  # [b]
            return pred

    return AlexNet


AlexNet_Sun = AlexNet_Sun_generator()

if __name__ == '__main__':
    from bacteria.code_sjh.utils.RamanData import Raman
    from bacteria.code_sjh.Core.basic_functions.fileReader import getRamanFromFile
    from bacteria.code_sjh.Core.basic_functions.visdom_func import startVisdomServer
    from bacteria.code_sjh.Core.basic_functions import visdom_func
    from scipy import interpolate
    import scipy.stats

    model_name = "waimiti_alexnet"

    startVisdomServer()
    vis = visdom.Visdom()
    test_db = Raman(dataroot, mode = "test", newfile = True,
                    LoadCsvFile = getRamanFromFile(wavelengthstart = 0, wavelengthend = 10000), )
    from torch.utils.data import DataLoader

    batchsz = len(test_db)
    label2name = test_db.label2name()
    loader = DataLoader(test_db, batch_size = batchsz, shuffle = True, num_workers = 0)
    sample_tensor, sample_label = test_db.__getitem__(1)
    sample_tensor = torch.unsqueeze(sample_tensor, dim = 0)

    alexnet = AlexNet_Sun(sample_tensor = sample_tensor, num_classes = 2)
    alexnet.set_model_name(model_name)

    save_dir = os.path.join(coderoot, "checkpoints", alexnet.model_name + ".mdl")
    alexnet.load(save_dir)
    alexnet.eval()
    spectrum_each_label = test_db.get_data_sorted_by_label()

    # attention map
    win_att = "win_att"
    for label in spectrum_each_label.keys():
        spectrum = spectrum_each_label[label]

        for pred_i in spectrum_each_label.keys():
            cam = alexnet.grad_cam(spectrum, pred_i)

            visdom_func.spectrum_vis(spectrum, test_db.wavelengths(),
                                     win = win_att + "_true_label_" + label2name[label] + "_CAM_label_" + label2name[
                                         pred_i], update = None,
                                     name = "spectrum")
            # (linespace_spectrum_xs,test_db.wavelengths()) ->插值 -> linespace_cam_xs -> cam_xs
            xs = numpy.linspace(400, 1800, len(test_db.wavelengths()))
            ys = test_db.wavelengths().np()
            xxs = numpy.linspace(400, 1800, cam.shape[-1])
            tck = interpolate.splrep(xs, ys)
            yys = interpolate.splev(xxs, tck)

            visdom_func.spectrum_vis(cam, yys,
                                     win = win_att + "_true_label_" + label2name[label] + "_CAM_label_" + label2name[
                                         pred_i], update = "append",
                                     name = "att")

    with torch.no_grad():
        for _, (sample_tensor, sample_label) in enumerate(loader):
            spectrum_each_label = alexnet.feature_vis(sample_tensor = sample_tensor, sample_label = sample_label,
                                                      vis = vis, update = None)

        pearsons = [[], []]
        mgs = []
        for l in range(16):
            s_0 = spectrum_each_label[0][:, l, :]  # [b,lenth]
            s_1 = spectrum_each_label[1][:, l, :]
            m_0, _, _ = visdom_func.data2mean_std(s_0)
            m_1, _, _ = visdom_func.data2mean_std(s_1)
            r, p = scipy.stats.pearsonr(m_0, m_1)
            pearsons[0].append(r)

            pearsons[1].append(p)

            stat, pval, mgcdict = scipy.stats.multiscale_graphcorr(s_0.np(), s_1.np(), is_twosamp = True)
            # print("channel-{}:r = {},p={},mgcdict".format(l,stat, pval))
            # print("channel-{}:r = {},p={}".format(l, r, p))
            mgs.append(stat)
            t_test = torch.Tensor(s_0.shape[-1])
            for idx in range(s_0.shape[-1]):
                r, t = scipy.stats.ttest_ind(s_0[:, idx].np(), s_1[:, idx].np(), equal_var = False)
                t_test[idx] = abs(r)
            win = "feature_vis_channel_" + str(l)
            xs = torch.linspace(400, 1800, s_0.shape[-1])
            vis.line(t_test, xs, win = win, update = "append", name = "ttest", opts = dict(
                title = win
            ))
        #
        # print(mgcdict)
        r = torch.tensor(pearsons[0])
        vis.bar(r, win = "pearsonr(x,y)", opts = dict(title = "pearsonr"))
        vis.bar(torch.tensor(mgs), win = "MGC", opts = dict(title = "MGS"))

        alexnet.dense_vis(vis = vis, update = "append")
