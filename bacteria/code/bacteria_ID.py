import glob
import os

import seaborn as sns
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.metrics import confusion_matrix
from scipy import sparse
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter

from tensorflow.python.keras.models import load_model

matplotlib.style.use("ggplot")


def baseline_remove_als(y, lam = 100000, p = 0.01, niter = 10):
	L = len(y)
	D = sparse.csc_matrix(np.diff(np.eye(L), 2))
	w = np.ones(L)
	for i in range(niter):
		W = sparse.spdiags(w, 0, L, L)
		Z = W + lam * D.dot(D.transpose())
		z = spsolve(Z, w * y)
		w = p * (y > z) + (1 - p) * (y < z)
		y = y - z  # subtract the background 'z' from the original data 'y'
	return y


# SG filter
def sg_filter(x, window_length = 11, polyorder = 3):
	x = savgol_filter(x, window_length, polyorder)
	return x


# spectral normalization
def norm_func(x, a = 0, b = 1):
	return ((b - a) * (x - min(x))) / (max(x) - min(x)) + a


def preprocess(x, y = None):
	x = np.apply_along_axis(baseline_remove_als, axis = 1, arr = x)
	# apply 'baseline_als' function to data x
	x = np.apply_along_axis(sg_filter, axis = 1, arr = x)
	# apply 'sg_filter' function to data x
	x = np.apply_along_axis(norm_func, axis = 1, arr = x)
	# apply 'norm_func' function to data x
	return x


filename = 'combined-p'  # 导入预处理后数据
data = {}
# data: [label][kind][number of line]; load path
for file in glob.glob("../../data/data_ID/**/combined-p.csv"):
	print("filepath=")
	print(file)
	path = file.split(os.path.sep)
	label = path[-2]
	kind = path[-1][:-4]
	if label not in data.keys():
		data[label] = {}
	data[label][kind] = pd.read_csv(file).values  # 直接读取预处理后数据

length = data['ab'][filename].shape[1]  # acquire spectral wavenumber range
classes = data.keys()
print('labels: ', data.keys())

train_data = {key: {} for key in data.keys()}
test_data = {key: {} for key in data.keys()}

# split data to train & test group
for key in data:
	for kind in data[key]:
		train_data[key][kind], test_data[key][kind] = train_test_split(data[key][kind], test_size = 0.3)


class DataGen(keras.utils.Sequence):
	def __init__(self, data, batch_size):
		self.data = data.copy()  # copy data from upper-layer
		self.batch_size = batch_size
		self.n_classes = len(data)  # len(data): number of the keys, 6 labels
		self.kinds = list(data[list(data.keys())[0]].keys())  # 1 kinds
		self.class_batch = self.batch_size // self.n_classes
		self.oh_enc = OneHotEncoder(sparse = False)
		self.oh_enc.fit(np.array(list(data.keys()))[:, np.newaxis])

	def __len__(self):
		length = []
		for key in self.data:  # data
			for kind in self.data[key]:
				length.append(self.data[key][kind].shape[0])
		return min(length) // self.class_batch

	def __getitem__(self, idx):
		samples = dict(zip(self.kinds, [[] for i in range(len(self.kinds))]))
		labels = []
		for key in self.data:
			for kind in self.kinds:
				sample = self.data[key][kind][idx * self.class_batch:(idx + 1) * self.class_batch]
				samples[kind].append(sample)

			labels.append(np.array([key] * self.class_batch))
		labels = np.concatenate(labels)[:, np.newaxis]
		samples = [np.concatenate(samples[key])[:, :, np.newaxis] for key in samples]
		return samples, self.oh_enc.transform(labels)

	def on_epoch_end(self):
		for key in self.data:
			for kind in self.data[key]:
				np.random.shuffle(self.data[key][kind])


class OnEpochEnd(tf.keras.callbacks.Callback):
	def __init__(self, callbacks):
		super().__init__()
		self.callbacks = callbacks

	def on_epoch_end(self, epoch, logs = None):
		for callback in self.callbacks:
			callback()


train_dg = DataGen(train_data, 60)
test_dg = DataGen(test_data, 60)
cust_callback = OnEpochEnd([train_dg.on_epoch_end, test_dg.on_epoch_end])

alpha = 0.15
drop = 0.6

inp_common = Input(shape = (length, 1))  # define input shape

l1 = Conv1D(filters = 8, kernel_size = 8, activation = 'linear')(inp_common)
l2 = BatchNormalization()(l1)
l3 = LeakyReLU(alpha)(l2)  # activation = Leaky ReLU
l4 = MaxPool1D(pool_size = 2)(l3)

l5 = Conv1D(filters = 16, kernel_size = 16, activation = 'linear')(l4)
l6 = BatchNormalization()(l5)
l7 = LeakyReLU(alpha)(l6)  # activation = Leaky ReLU
l8 = MaxPool1D(pool_size = 2, name = "output")(l7)

out_common = l8

common_conv = Model(inp_common, out_common)  # common model

inp1 = Input(shape = (length, 1), name = "input")

out1 = common_conv(inp1)

x0 = out1  # concatenate 数组拼接
x1 = Flatten()(x0)
x2 = Dense(64, activation = 'linear')(x1)  # 全连接层
x3 = LeakyReLU(alpha)(x2)  # activation = Leaky ReLu
x4 = Dropout(drop)(x3)

x5 = Dense(32, activation = 'linear')(x4)
x6 = LeakyReLU(alpha)(x5)
x7 = Dropout(drop)(x6)

out = Dense(len(data), activation = 'softmax', name = "main_output")(x7)  # 输出层 核的个数 = len(data)
# softmax 用于多分类

model = Model(inputs = [inp1], outputs = [out])

model.compile(optimizer = "adam", loss = "categorical_crossentropy",
              metrics = ['accuracy'])

History = model.fit(train_dg, epochs = 100, validation_data = test_dg, callbacks = [cust_callback])

# 测试集数据扩增
X = []
Y = []
augment = 100  # 测试集数据随机扩增的倍数
for i in range(augment):
	for x, y in test_dg:
		X.append(x)
		Y.append(y)
	test_dg.on_epoch_end()
X = [arr for arr in np.concatenate(X, axis = 1)]
Y = np.concatenate(Y)

Y_prob = model.predict(X)  # array of probability
Y_pred = (Y_prob > 0.5).astype(int)  # array of predicted labels
print(classification_report(Y, Y_pred, target_names = train_dg.oh_enc.categories_[0], digits = 3))

model.save('model-ID.h5')

# 获取混淆矩阵
cm = confusion_matrix(Y.argmax(axis = 1), Y_pred.argmax(axis = 1))
cm = [i / augment for i in cm]  # 恢复原始测试集，除以扩增倍数


# save & plot the confusion matrix
def plot_confusion_matrix(cm):
	sns.set()
	fig, ax = plt.subplots()
	sns.heatmap(cm, annot = True, fmt = 'g', ax = ax, cmap = "YlGnBu")  # Colors: YlGnBu
	ax.set_xlabel('Predicted Label')
	ax.set_ylabel('Actual Label')
	plt.savefig('./confusion_matrix_ID.png')
	plt.show()


# plot the figure of loss func and acc
def plot_training_history(History):
	acc = History.history['accuracy']
	val_acc = History.history['val_accuracy']
	loss = History.history['loss']
	val_loss = History.history['val_loss']
	epochs = range(len(acc))

	plt.figure(1)

	plt.subplot(121)
	plt.plot(epochs, acc, 'r', label = 'Training accuracy')
	plt.plot(epochs, val_acc, 'b', label = 'validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend(loc = 'lower right')

	plt.subplot(122)
	plt.plot(epochs, loss, 'r', label = 'Training loss')
	plt.plot(epochs, val_loss, 'b', label = 'validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()


plot_confusion_matrix(cm)

plot_training_history(History)

print('well done')
