# %%
import glob
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch.autograd import Variable
import seaborn as sns

# %%
filename = 'combined-p'  # 导入预处理后数据
data = {}
label = []
# data: [label][kind][number of line]; load path
for file in glob.glob("E:/305/pre_100-dubbed/**/combined-p.csv"):
    path = file.split(os.path.sep)
    print(path)
    label.append(path[-2])
    if path[-2] not in data.keys():
        data[path[-2]] = pd.read_csv(file, header=None).values  # 直接读取预处理后数据->data
Y_spectral = np.zeros((len(label) * 100,))
index = 0
X_spectral = []
for i in label:
    if index == 0:
        X_spectral = data[i]
    else:
        X_spectral = np.concatenate((X_spectral, data[i]), axis=0)  # 所有光谱叠加
    Y_spectral[index * 100:(index + 1) * 100] = label.index(i)  # 添加标签
    index = index + 1

# newClass = {
#     'aba': ['31', '253', '48', '49'],
#     'sep': ['72', '17', '51', '78'],
#     'eco': ['41', '53-', '58', '59'],
#     'ppu': ['15', '81', '82'],
#     'kpn': ['39', '42', '45'],
#     'efa': ['25', '62', '63', '64'],
#     'sau': ['28', '18', '47'],
#     'shl': ['7', '22', '53', '54'],
#     'efm': ['26', '27', '84', '85'],
#     'pma': ['5', '37', '75', '55'],
#     'pae': ['33', '151', '68', '67'],
#     'sgc': ['2', '29'],
#     'ecl': ['57', '79', '1', '56'],
#     'sma': ['24', '34', '35', '46']
# }

newClass = {
      'G—': ['31', '253', '48', '49', '52-', '53-', '58', '59',
             '15', '81', '82', '45', '39', '42', '5', '55','75',
             '76', '33', '151', '66', '67', '1', '56', '57', '79',
             '34', '24', '36', '46','30','41','83','38','37','68','40','34'],
      'G+': ['17', '51', '72', '16', '25', '62', '63', '64', '28',
             '18', '47', '7', '52', '53', '54', '26', '21', '84',
             '85', '2', '29','16','65','19','22','27','77']
}

# %%
def combined_same(data, newClass):  # 传入光谱数据字典，传入新分类方案
    newData = {}
    for i in newClass.keys():
        for j in newClass[i]:
            if newClass[i].index(j) == 0:
                newData[i] = data[j]
            else:
                newData[i] = np.concatenate((newData[i], data[j]), axis=0)
    return newData


newData = combined_same(data, newClass)  # 结合上述分类方案更新合并后的数据
Y_label = {}
finetuning_p = 0.2
num = 0
for i in newData:
    Y_label[i] = len(newData[i])
    if num == 0:
        iTrain = np.arange(Y_label[i])
        np.random.shuffle(iTrain)
        X_finetuning_idx = iTrain[:int(Y_label[i] * finetuning_p)]
        X_train_idx = iTrain[int(Y_label[i] * finetuning_p):]

        X_spectral_FT = newData[i][X_finetuning_idx]  # 微调数据集
        X_spectral = newData[i][X_train_idx]  # 训练验证数据集

        Y_spectral_FT = np.zeros(int(Y_label[i] * finetuning_p), )
        Y_spectral = np.zeros(int(Y_label[i] * (1 - finetuning_p)), )
    else:
        iTrain = np.arange(Y_label[i])
        np.random.shuffle(iTrain)
        X_finetuning_idx = iTrain[:int(Y_label[i] * finetuning_p)]
        X_train_idx = iTrain[int(Y_label[i] * finetuning_p):]

        X_spectral_FT = np.concatenate((X_spectral_FT, newData[i][X_finetuning_idx]), axis=0)  # 微调光谱叠加
        Y_spectral_FT = np.concatenate((Y_spectral_FT, np.ones((int(Y_label[i] * finetuning_p)), ) * num))

        X_spectral = np.concatenate((X_spectral, newData[i][X_train_idx]), axis=0)  # 训练光谱叠加
        Y_spectral = np.concatenate((Y_spectral, np.ones((int(Y_label[i] * (1 - finetuning_p))), ) * num))
    num = num + 1
# 每一种
X_spectral = np.concatenate((X_spectral, X_spectral), axis=0)  # 所有光谱叠加
Y_spectral = np.concatenate((Y_spectral, Y_spectral))

# %%
dl_tr, dl_val, dl_test = spectral_dataloaders(X_spectral, Y_spectral, batch_size=10)
torch.save(dl_val, 'dl_val.pth')


# %%

# class RNN_Model(nn.Module):
#     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#         super(RNN_Model, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.layer_dim = layer_dim
#         self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         # (layer_dim,batch_size,hidden_size)
#         h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
#         out, hn = self.rnn(x, h0.detach())
#         out = self.fc(out[:, -1, :])
#         return out


# %%
class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim,dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # (layer_dim,batch_size,hidden_size)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


# %%
input_dim = 528
hidden_dim = 128
layer_dim = 3
output_dim = 14
# len(newClass)

device = torch.device('cpu' if torch.cuda.is_available() else 'cuda:0')
model = LSTM_Model(input_dim, hidden_dim, layer_dim, output_dim).to(device)

# %%
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(len(list(model.parameters())))

# %%
for i in range(10):
    print('参数：%d' % (i + 1))
    print(list(model.parameters())[i].size())

# %%
sequence_dim = 528
Loss_tr = []
Loss_val = []
epoch_tr = []
accuracy_tr = []
accuracy_val = []
# iteration_list = []
epoch_list = []

EPOCHS = 200
iteration = 0
for epoch in range(EPOCHS):
    correct_tr = 0.0
    total_tr = 0.0
    for i, (spectrals, labels) in enumerate(dl_tr):
        model.train()
        spectrals = spectrals.requires_grad_().to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(spectrals)
        predict_tr = torch.max(outputs.data, 1)[1]
        total_tr += labels.size(0)
        if torch.cuda.is_available():
            correct_tr += (predict_tr.cuda() == labels.cuda()).sum()
        else:
            correct_tr += (predict_tr == labels).sum()
        loss_tr = criterion(outputs, labels.long())
        loss_tr.backward()
        optimizer.step()
        iteration += 1
        if iteration % 1000 == 0:
            model.eval()
            correct_val = 0.0
            total_val = 0.0

            for spectrals, labels in dl_val:
                spectrals.to(device)
                outputs = model(spectrals)
                loss_val = criterion(outputs, labels.long())
                predict_val = torch.max(outputs.data, 1)[1]
                total_val += labels.size(0)
                if torch.cuda.is_available():
                    correct_val += (predict_val.cuda() == labels.cuda()).sum()
                else:
                    correct_val += (predict_val == labels).sum()

            accuracy_v = correct_val / total_val * 100
            Loss_val.append(loss_val.data)
            accuracy_val.append(accuracy_v)
            # iteration_list.append(iteration)
            epoch_list.append(epoch)
            print("iteration:{},epoch:{},Loss_val:{},Accuracy_val:{}".format(iteration, epoch, loss_val.item(),
                                                                             accuracy_v))
    Loss_tr.append(loss_tr.data)
    accuracy_t = correct_tr / total_tr * 100
    accuracy_tr.append(accuracy_t)
    epoch_tr.append(epoch)
    print("epoch:{},Loss_tr:{},Accuracy_tr:{}".format(epoch, loss_tr.item(), accuracy_t))

# %%
LSTM_PATH = 'Layer_2_output_14_2.ckpt'
torch.save(model.state_dict(), LSTM_PATH)
# %%
epoch_list_cpu = torch.tensor(epoch_list, device='cpu')
# iteration_list_cpu = torch.tensor(iteration_list, device='cpu')
loss_list_val = torch.tensor(Loss_val, device='cpu')
accuracy_list_val = torch.tensor(accuracy_val, device='cpu')
loss_list_tr = torch.tensor(Loss_tr, device='cpu')
accuracy_list_tr = torch.tensor(accuracy_tr, device='cpu')
epoch_list_tr = torch.tensor(epoch_tr, device='cpu')


