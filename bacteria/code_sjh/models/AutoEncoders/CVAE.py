import csv
import time

import numpy
import torch
import visdom
from torch import nn

from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
# 记录各个路径
import torch.nn.functional as F
import os, sys
from sklearn.preprocessing import LabelBinarizer

coderoot = os.path.split(os.path.split(__file__)[0])[0]
coderoot = os.path.split(coderoot)[0]
checkpointroot = os.path.join(coderoot, "checkpoints")
projectroot = os.path.split(coderoot)[0]

try:
    from ..BasicModule import BasicModule, Flat
    from ...utils import RamanData, visdom_utils
    from ...utils import Validation as validation
    from .VAE import VAE
    from ..Criteons import compute_gassian_kl
    from ..Parts import _CNN_layer, _TCNN_layer
except:
    sys.path.append(coderoot)
    from models.BasicModule import BasicModule, Flat
    from utils import RamanData
    from utils.Validation import visdom_utils
    from utils import Validation as validation
    from models.AutoEncoders.VAE import VAE
    from models.Criteons import compute_gassian_kl
    from models.Parts import _CNN_layer, _TCNN_layer


class _CVAE_CNN_encoder4(BasicModule):
    def __init__(self, sample_tensor,
                 num_classes,
                 neck_axis = 2,
                 dropout = 0.1,
                 kernelsize = 8,
                 verbose = False,
                 stride = 3):
        super(_CVAE_CNN_encoder4, self).__init__()
        self.num_classes = num_classes
        self.CNN1 = nn.Sequential(
            # self, ch_in, ch_out, dropout = 0.1, kernelsize = 8, stride = 3, Maxpool = True, verbose = False
            _CNN_layer(1, 32, dropout = dropout, stride = stride, kernelsize = kernelsize, verbose = verbose),
            _CNN_layer(32, 32, dropout = dropout, stride = stride, kernelsize = kernelsize, verbose = verbose),
            _CNN_layer(32, 64, dropout = dropout, stride = stride, kernelsize = kernelsize, verbose = verbose),
            _CNN_layer(64, 64, dropout = dropout, stride = stride, kernelsize = kernelsize, Maxpool = False,
                       verbose = verbose),
            Flat(),
        )

        with torch.no_grad():
            sample_tensor1 = self.CNN1(sample_tensor)
            self.CNN_out_lenth = sample_tensor1.shape[-1]
        self.Dense1 = nn.Sequential(
            nn.Linear(self.CNN_out_lenth, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.Dense2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.Dense3 = nn.Sequential(
            nn.Linear(32, neck_axis),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.CNN1(x)  # [b,cnn_out_length]
        out = self.Dense1(out)
        out = self.Dense2(out)
        out = self.Dense3(out)
        return out  # [b,neck_axis]


class _CVAE_TCNN_decoder4(BasicModule):
    def __init__(self, sample_tensor,
                 num_classes,
                 TCNN_in_lenth,
                 neck_axis = 2,
                 kernelsize = 8,
                 stride = 3,
                 dropout = 0.1,
                 verbose = False):
        super(_CVAE_TCNN_decoder4, self).__init__()
        assert TCNN_in_lenth % 64 == 0, "CNN_in_lenth mod 64 must be 0"
        self.num_classes = num_classes
        self.lenth = sample_tensor.shape[-1]
        self.channel = sample_tensor.shape[-2]
        self.TCNN_in_lenth = TCNN_in_lenth
        self.classifier = nn.Sequential(
            nn.Linear(neck_axis, num_classes),
            nn.Sigmoid(), )
        self.Dense = nn.Sequential(
            nn.Linear(neck_axis, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, TCNN_in_lenth),
            # nn.Linear(128,TCNN_in_lenth+num_classes),
            nn.ReLU(),
            nn.Dropout(dropout)

            # nn.Linear(neck_axis,TCNN_in_lenth)
        )  # [b,in_l]

        self.TCNN = nn.Sequential(
            # self, ch_in, ch_out, dropout = 0.1, kernelsize = 8, stride = 3, Maxpool = True, verbose = False
            _TCNN_layer(64, 64, dropout = dropout, kernel_size = kernelsize, stride = stride, verbose = verbose),
            # [b,64,l2]
            _TCNN_layer(64, 32, dropout = dropout, kernel_size = kernelsize, stride = stride, verbose = verbose),
            # [b,32,l3]
            _TCNN_layer(32, 32, dropout = dropout, kernel_size = kernelsize, stride = stride, verbose = verbose),
            # [b,32,l4]
            _TCNN_layer(32, 1, dropout = dropout, kernel_size = kernelsize, stride = stride, verbose = verbose),
            # [b,1,l5]
            nn.Sequential(
                nn.Conv1d(1, 1, kernel_size = kernelsize, stride = 1, ),
                nn.ReLU(),
                # nn.Sigmoid(),
                nn.Tanh(),
            )
        )

    def forward(self, x):
        out = self.Dense(x)  # [b,in_l+n_c]

        # out, y_c_hat = out.split([self.TCNN_in_lenth, self.num_classes], dim = 1)  # [b,in_l],[b,n_c]
        y_c_hat = self.classifier(x)
        out = out.view(-1, 64, self.TCNN_in_lenth // 64)  # [b,64,in_l//64]

        x_hat = self.TCNN(out)  # [b,l]

        assert x_hat.shape[-1] == self.lenth, \
            " The length of the tensor({}) is not available, maybe u can reshape it to {}?" \
                .format(self.lenth, x_hat.shape[-1])
        return x_hat, y_c_hat


class _CVAE_CNN_encoder2(BasicModule):
    def __init__(self, sample_tensor,
                 num_classes,
                 neck_axis = 2,
                 dropout = 0.1,
                 kernelsize = 8,
                 verbose = False,
                 stride = 3):
        super(_CVAE_CNN_encoder2, self).__init__()
        self.num_classes = num_classes
        self.CNN1 = nn.Sequential(
            # self, ch_in, ch_out, dropout = 0.1, kernelsize = 8, stride = 3, Maxpool = True, verbose = False
            _CNN_layer(sample_tensor.shape[-2], 16, dropout = dropout, stride = stride, kernelsize = kernelsize, verbose = verbose),
            _CNN_layer(16, 64, dropout = dropout, stride = stride, kernelsize = kernelsize, verbose = verbose),
            Flat(),
        )

        with torch.no_grad():
            sample_tensor1 = self.CNN1(sample_tensor)
            self.CNN_out_lenth = sample_tensor1.shape[-1]
        self.Dense1 = nn.Sequential(
            nn.Linear(self.CNN_out_lenth, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.Dense2 = nn.Sequential(
            nn.Linear(128, neck_axis),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.CNN1(x)  # [b,cnn_out_length]
        out = self.Dense1(out)
        out = self.Dense2(out)
        return out  # [b,neck_axis]


class _CVAE_TCNN_decoder2(BasicModule):
    def __init__(self, sample_tensor,
                 num_classes,
                 TCNN_in_lenth,
                 neck_axis = 2,
                 kernelsize = 8,
                 stride = 3,
                 dropout = 0.1,
                 verbose = False):
        super(_CVAE_TCNN_decoder2, self).__init__()
        assert TCNN_in_lenth % 64 == 0, "CNN_in_lenth mod 64 must be 0"
        self.num_classes = num_classes
        self.lenth = sample_tensor.shape[-1]
        self.channel = sample_tensor.shape[-2]
        self.TCNN_in_lenth = TCNN_in_lenth
        self.classifier = nn.Sequential(
            nn.Linear(neck_axis, num_classes),
            nn.Sigmoid(), )
        self.Dense = nn.Sequential(
            nn.Linear(neck_axis, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, TCNN_in_lenth),
            # nn.Linear(128,TCNN_in_lenth+num_classes),
            nn.ReLU(),
            nn.Dropout(dropout)

            # nn.Linear(neck_axis,TCNN_in_lenth)
        )  # [b,in_l]

        self.TCNN = nn.Sequential(
            # self, ch_in, ch_out, dropout = 0.1, kernelsize = 8, stride = 3, Maxpool = True, verbose = False
            _TCNN_layer(64, 16, dropout = dropout, kernel_size = kernelsize, stride = stride, verbose = verbose),
            # [b,32,l3]
            _TCNN_layer(16, sample_tensor.shape[-2], dropout = dropout, kernel_size = kernelsize, stride = stride, verbose = verbose),
            # [b,1,l5]
            nn.Sequential(
                nn.Conv1d(1, 1, kernel_size = kernelsize, stride = 1, ),
                nn.ReLU(),
                # nn.Sigmoid(),
                nn.Tanh(),
            )
        )

    def forward(self, x):
        out = self.Dense(x)  # [b,in_l+n_c]

        # out, y_c_hat = out.split([self.TCNN_in_lenth, self.num_classes], dim = 1)  # [b,in_l],[b,n_c]
        y_c_hat = self.classifier(x)
        out = out.view(-1, 64, self.TCNN_in_lenth // 64)  # [b,64,in_l//64]

        x_hat = self.TCNN(out)  # [b,l]

        assert x_hat.shape[-1] == self.lenth, \
            " The length of the tensor({}) is not available, maybe u can reshape it to {}?" \
                .format(self.lenth, x_hat.shape[-1])
        return x_hat, y_c_hat


class CVAE(VAE):
    def __init__(self, num_classes):
        super(CVAE, self).__init__()
        self.num_classes = num_classes
        self.lb = LabelBinarizer()
        self.lb.fit(list(range(0, self.num_classes)))

    def onehot(self, y, device = None):  # y：[b]
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y_n = y.numpy()
        if self.num_classes > 2:
            y_one_hot = self.lb.transform(y_n)
        else:
            y_one_hot = [[0, 0]] * len(y_n)
            for i in range(len(y_n)):
                y_one_hot[i][y_n[i]] = 1
            y_one_hot = numpy.array(y_one_hot)
        floatTensor = torch.FloatTensor(y_one_hot)  # y:[b,n_c]
        return floatTensor.to(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):  # y为label,x:[b,1,l]
        if y is None:  # 不输入y时作为VAE处理
            pass
        elif len(y.shape) == 1:
            y = self.onehot(y)  # y_c:[b,n_c]

        mu_N = 0 if y is None else self.dense_label(y)

        h_ = self.Encoder(x)
        mu, sigma = h_.chunk(2, dim = 1)  # [b neck_axis]->[b neck_axis/2]*2
        # sigma = F.relu(sigma)
        h = self.reparameterize(mu, sigma)

        x_hat, y_c_hat = self.Decoder(h)
        x_hat = x_hat.view(x.shape)  # 形状还原

        kld = torch.sum(compute_gassian_kl(mu, sigma, mu_N, 1)) / numpy.prod(x.shape)

        return x_hat, y_c_hat, kld


class CVAE_Dlabel_Dclassifier(CVAE):
    def __init__(self, sample_tensor, num_classes, neck_axis = 2, dropout = 0.3, kernelsize = 8, stride = 3):
        super(CVAE_Dlabel_Dclassifier, self).__init__(num_classes = num_classes)
        self.neck_axis = neck_axis
        self.model_name = "CVAE"
        self.dense_label = nn.Sequential(
            nn.Linear(self.num_classes, self.neck_axis),
            nn.Sigmoid()
        )
        self.Encoder = _CVAE_CNN_encoder4(sample_tensor = sample_tensor, num_classes = num_classes,
                                          neck_axis = neck_axis * 2, dropout = dropout,
                                          kernelsize = kernelsize, stride = stride)
        self.Decoder = _CVAE_TCNN_decoder4(sample_tensor = sample_tensor, num_classes = num_classes,
                                           TCNN_in_lenth = self.Encoder.CNN_out_lenth,
                                           neck_axis = neck_axis, dropout = dropout,
                                           kernelsize = kernelsize, stride = stride)

class CVAE2_Dlabel_Dclassifier(CVAE):
    def __init__(self, sample_tensor, num_classes, neck_axis = 2, dropout = 0.3, kernelsize = 8, stride = 3):
        super(CVAE2_Dlabel_Dclassifier, self).__init__(num_classes = num_classes)
        self.neck_axis = neck_axis
        self.model_name = "CVAE"
        self.dense_label = nn.Sequential(
            nn.Linear(self.num_classes, self.neck_axis),
            # nn.Sigmoid()
        )
        self.Encoder = _CVAE_CNN_encoder2(sample_tensor = sample_tensor, num_classes = num_classes,
                                          neck_axis = neck_axis * 2, dropout = dropout,
                                          kernelsize = kernelsize, stride = stride)
        self.Decoder = _CVAE_TCNN_decoder2(sample_tensor = sample_tensor, num_classes = num_classes,
                                           TCNN_in_lenth = self.Encoder.CNN_out_lenth,
                                           neck_axis = neck_axis, dropout = dropout,
                                           kernelsize = kernelsize, stride = stride)






