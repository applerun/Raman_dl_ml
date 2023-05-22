# coding:utf8
import os.path
import warnings

import torch as t
import time

import torch.nn


class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self, sample_tensor = None):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 默认名字
        self.model_loaded = False
        self.sample_tensor = sample_tensor.clone()

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(t.load(path))
        self.model_loaded = True

    def save(self, name = None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if not os.path.isdir(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))
        if name is None:
            prefix = '../checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name

    def save_model(self, name = None, sample_tensor = None, opset_version = 11):
        if not os.path.isdir(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))
        if name is None:
            prefix = '../saved_models/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        if sample_tensor is None:
            sample_tensor = self.sample_tensor
        if sample_tensor is None:
            warnings.warn("no sample tensor provided, cannot generate graph")
        torch.onnx.export(self, self.sample_tensor, name, opset_version = opset_version,
                          input_names = ['input'],
                          output_names = ['output'])

    def set_model_name(self, name):
        self.model_name = name


class Flat(t.nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flat, self).__init__()
        # self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)
