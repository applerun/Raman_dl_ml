import glob
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from bacteria.code_sjh.models.BasicModule import BasicModule
from bacteria.code_sjh.utils.RamanData import pytorchlize

class LSTM_Model(BasicModule):
    def __init__(self,
                 sample_tensor: torch.Tensor, num_classes,

                 hidden_dim = 128,
                 layer_dim = 3,
                  ):
        sample_tensor = pytorchlize(sample_tensor)
        input_dim = 528,
        output_dim = num_classes
        super(LSTM_Model, self).__init__(sample_tensor = sample_tensor)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, dropout = 0.5, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.model_name = "ResNet18"

    def forward(self, x):
        # (layer_dim,batch_size,hidden_size)
        device = next(self.parameters()).device
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out
