import copy

import numpy

from bacteria.code_sjh.models.BasicModule import BasicModule, Flat
import torch.nn as nn
import torch
import numpy as np

"""
Goodfellow I J, Pouget-Abadie J, Mirza M, et al. 
Generative Adversarial Nets; 
proceedings of the 28th Conference on Neural Information Processing Systems (NIPS), Montreal, CANADA, F 2014 Dec 08-13, 
2014 [C]. 2014.]]

生成对抗网络（GANs）
包含一个估计生成模型的框架，可以直接从的底层数据分布中抽取样本，而不需要明确地定义一个概率分布。
它由两个模型组成：一个生成器G和一个鉴别器D。
    生成模型G以一个从先验分布Pz (z)中采样的随机噪声向量z作为输入，Pz通常是高斯分布或均匀分布，然后将z映射到数据空间为G（z，θg），
其中G是一个参数为θg的神经网络。
    被表示为G (z)或xg的假样本被期望与真实的样本相似。鉴别器是由θd参数化的第二个神经网络，输出一个样本出现的概率D（x，θd）。
    对判别网络D进行优化，使分配假数据和真数据标签时，概率对数似然性最大化，而生成模型G则经过训练，使D出错的对数可能性最大化。
    通过对抗性的过程，G逐步估计底层数据的概率分布，并生成真实的样本。
"""


class GeneratorDense(BasicModule):
    default_dense_node = [128, 256, 512, 1024]

    def __init__(self, sample_tensor: torch.Tensor or numpy.ndarray, len_z: int, dense_node = None):
        """

        @param sample_tensor: 样例张量，shape为[b,c,l] 或[b,l]
        @param len_z:
        """
        super(GeneratorDense, self).__init__()
        self.raman_shape = sample_tensor.shape[1:]
        layers = []
        if dense_node is None:
            dense_node = copy.deepcopy(self.default_dense_node)
        dense_node = [len_z] + dense_node
        for i in range(len(dense_node)-1):
            layers += self.block(dense_node[i], dense_node[i + 1], normalize = False if i == 0 else True)
        self.model = nn.Sequential(
            *layers,
            nn.Linear(1024, int(np.prod(self.raman_shape))),
            nn.Tanh()
        )

    def block(self, in_feat, out_feat, normalize = True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace = True))
        return layers

    def forward(self, z):  # z [b,l]
        raman = self.model(z)
        raman = raman.view(raman.shape[0], *self.raman_shape)
        return raman  # raman [b,c,l]


class DiscriminatorDense(BasicModule):
    default_dense_node = [512, 256, 1]

    def __init__(self, sample_tensor: torch.Tensor, dense_node = None):
        super(DiscriminatorDense, self).__init__()

        if dense_node is None:
            dense_node = copy.deepcopy(self.default_dense_node)
        node_list = [int(np.prod(sample_tensor.shape[-2:]))] + dense_node
        blocks = []
        for i in range(len(node_list) - 1):
            blocks.append(self.block(node_list[i], node_list[i + 1],
                                     activate = nn.LeakyReLU if i < len(node_list) - 1 else nn.Sigmoid()))
        self.model = nn.Sequential(
            Flat(),
            *blocks
        )

    def block(self, in_feat, out_feat, activate = nn.LeakyReLU, act_args = None, act_kwargs = None):
        """
        @param in_feat:
        @param out_feat:
        @param activate:
        @param act_kwargs:
        @param act_args:
        @return:
        """

        if act_args is None:
            act_args = []
        if act_kwargs is None:
            act_kwargs = {}
        layers = [nn.Linear(in_feat, out_feat)]
        if activate is not None:
            layers.append(activate(*act_args, **act_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    true_tensor = torch.zeros((16, 1, 854))
    z = torch.randn((16, 100))

    G = GeneratorDense(true_tensor, 100)
    print(G)
    D = DiscriminatorDense(true_tensor)
    fake_tensor = G(z)
    dis_true = D(true_tensor)
    dis_false = D(fake_tensor)
    pass
