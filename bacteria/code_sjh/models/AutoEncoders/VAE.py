import copy
import torch
import visdom
import numpy
import os, sys
from torch import nn

coderoot = os.path.split(os.path.split(__file__)[0])[0]
coderoot = os.path.split(coderoot)[0]

checkpointroot = os.path.join(coderoot, "checkpoints")
projectroot = os.path.split(coderoot)[0]
dataroot = os.path.join(projectroot, "data", "data_AST")
try:
    from ..BasicModule import BasicModule, Flat
    from ...utils import RamanData
    from .AutoEncoder import AutoEncoder
    from bacteria.code_sjh.models.Parts import _CNN_encoder4, _TCNN_decoder4, _dense_decoder, _dense_encoder
except:
    sys.path.append(coderoot)
    from models.BasicModule import BasicModule, Flat
    from utils import RamanData
    from models.AutoEncoders.AutoEncoder import AutoEncoder
    from bacteria.code_sjh.models.Parts import _CNN_encoder4, _TCNN_decoder4, _dense_decoder, _dense_encoder


class VAE(AutoEncoder):
    def __init__(self):
        super(VAE, self).__init__()

    def forward(self, x):
        shape = x.shape
        h_ = self.Encoder(x)
        mu, sigma = h_.chunk(2, dim = 1)  # [b neck_axis]->[b neck_axis/2]*2
        h = self.reparameterize(mu, sigma)
        x_hat = self.Decoder(h)
        x_hat = x_hat.view(*shape)  # 形状还原

        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-10 + torch.pow(sigma, 2)) -
            1
        ) / numpy.prod(shape)

        return x_hat, kld

    def encode(self, x, rand = .0):
        h_ = self.Encoder(x)
        mu, sigma = h_.chunk(2, dim = 1)  # [b neck_axis]->[b neck_axis/2]*2
        h = self.reparameterize(mu, sigma, rand = rand)
        return h

    def reparameterize(self, mu, std, rand = .0):
        # 训练时使用重参数化技巧，测试时不用。（测试时应该可以用）
        eps = torch.randn_like(std)
        if self.training:
            return eps * std + mu
        else:
            return eps * rand + mu

    def neck_vis(self, x, label, update = None, vis = None, markersymbol = 'cross', markersize = 4, win = 'neck vis',
                 repeat = 0, rand = 0.5):
        """

        :param x: [l] or [b,l] or [b,c=1,l] one or a batch of spectrum of the same label
        :param label: the label of this batch of data (used as the legend of the dots)
        :param update: None if to rewrite and "append" if to add
        :param vis: visdom.Visdom, a new one will be created if None
        :param markersymbol: marker symbol of the dots
        :param markersize: marker size of the dots
        :param win: the window name in visdom localhost and the title of
         the window
        :return: encode
        """
        self.eval()
        assert self.neck_axis < 4, "too many neck axis"
        if not self.model_loaded:
            self.load(os.path.join(coderoot, "checkpoints", self.model_name + ".mdl"))
        if vis is None:
            vis = visdom.Visdom()
        x = RamanData.pytorchlize(x)
        with torch.no_grad():
            h = self.encode(x, rand = rand)
            for r in range(repeat):
                h_t = self.encode(x, rand = rand)
                h = torch.cat((h, h_t))

        if not type(label) == str:
            label = str(label)
        vis.scatter(h,
                    win = win,
                    update = update,
                    name = label,
                    opts = dict(
                        title = win,
                        markersize = markersize,
                        showlegend = True,
                        markersymbol = markersymbol,
                    )
                    )
        return x


class VAE_conv4(VAE):
    def __init__(self, sample_tensor, neck_axis = 2, dropout = 0.3, kernelsize = 8, stride = 3):
        super(VAE_conv4, self).__init__()
        self.neck_axis = neck_axis
        self.model_name = "convolutional_auto_encoder"
        self.Encoder = _CNN_encoder4(copy.deepcopy(sample_tensor), neck_axis = neck_axis * 2, dropout = dropout,
                                     kernelsize = kernelsize, stride = stride)
        self.Decoder = _TCNN_decoder4(sample_tensor, TCNN_in_lenth = self.Encoder.CNN_out_lenth, neck_axis = neck_axis,
                                      dropout = dropout,
                                      kernelsize = kernelsize, stride = stride)


class VAE_dense(VAE):
    def __init__(self, sample_tensor: torch.Tensor, neck_axis = 10, dropout = 0.1, bias = True):
        super(VAE_dense, self).__init__()
        self.sample_tensor: torch.Tensor = sample_tensor.clone()
        self.model_name = "VAE_dense"
        self.Flat = Flat()
        with torch.no_grad():
            t = self.Flat(self.sample_tensor)
        self.lenth = t.shape[1]
        self.Encoder = _dense_encoder
        self.Decoder = _dense_decoder


class muted_VAEncoder():
    def __init__(self, module: BasicModule, mdlfile):
        self.encoder = module.Encoder
        self.encoder.load(mdlfile)
        self.encoder.eval()
        return

    def __call__(self, input: numpy.ndarray):
        x = torch.tensor(input).to(torch.float32)
        x = RamanData.pytorchlize(x)

        h_: torch.Tensor = self.encoder(x)
        mu, sigma = h_.chunk(2, dim = 1)  # [b neck_axis]->[b neck_axis/2]*2
        h = mu + sigma * torch.randn_like(sigma)
        out = torch.squeeze(h)

        out = out.detach().np()

        return out
