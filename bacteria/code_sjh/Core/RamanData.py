import os
from torch.utils.data import Dataset
import csv
import numpy
import random
import torch
from bacteria.code_sjh.Core.basic_functions.data_functions import *
from bacteria.code_sjh.Core.basic_functions.fileReader import *
from bacteria.code_sjh.Core.basic_functions import visdom_func


class RamanDatasetCore(Dataset):  # 增加了一些基础的DataSet功能
    def __init__(self,
                 dataroot: str,
                 mode = "train",
                 t_v_t = None,
                 sfpath = "Ramans.csv",
                 shuffle = True,
                 transform = None,
                 LoadCsvFile = None,
                 backEnd = ".csv",
                 unsupervised: bool = False,
                 noising = None,
                 newfile = False,
                 k_split: int = None,
                 k: int = 0,
                 ratio: dict = None,
                 ):
        """

        :param dataroot: 数据的根目录
        :param resize: 光谱长度(未实装)
        :param mode: "train":训练集 "val":验证集 "test":测试集
        :param t_v_t:[float,float,float] 分割所有数据train-validation-test的比例
        :param sfpath: 数据文件的名称，初始化时，会在数据根目录创建记录数据的csv文件，文件格式：label，*spectrum，如果已经有该记录文件
        :param shuffle: 是否将读取的数据打乱
        :param transform: 数据预处理/增强
        :param LoadCsvFile:根据数据存储格式自定义的读取文件数据的函数
        combined:生成器，第一个为光谱数据的header
        :param backEnd:str 存储文件的后缀
        :param supervised: 如果为无监督学习，将noising前的信号设置为label
        :param noising: callable——input：1d spectrum output：noised 1d spectrum

        """

        # assert mode in ["train", "val", "test"]
        assert os.path.isdir(dataroot), "dataroot {} do not exist".format(dataroot)
        super(RamanDatasetCore, self).__init__()

        if t_v_t is None and k_split is None:  # 分割train-validation-test
            t_v_t = [0.7, 0.2, 0.1]
        # if type(t_v_t) is list:
        # 	t_v_t = numpy.array(t_v_t)
        if mode == "all":
            mode = "train"
            t_v_t = [1, 0, 0]
            k_split = None
        self.k_split = k_split
        if k_split is not None:  # k_split 模式
            # t_v_t = [x*k_split for x in t_v_t]
            assert 0 <= k < k_split, "k must be in range [{},{}]".format(0, k_split - 1)
        self.k = k
        # assert t_v_t[0] + t_v_t[1] <= 1
        self.tvt = t_v_t
        self.new = newfile

        self.LoadCsvFile = LoadCsvFile
        self.root = dataroot

        self.name2label = {}  # 为每个分类创建一个label
        self.sfpath = sfpath
        self.shuff = shuffle
        self.mode = mode
        self.dataEnd = backEnd
        self.transform = transform
        self.unsupervised = unsupervised
        self.noising = noising

        self.train_split = self.tvt[0]
        self.validation_split = self.tvt[1]
        self.test_split = self.tvt[2]
        self.xs = None
        self.RamanFiles = []
        self.labels = []
        self.Ramans = []
        self.ratio = ratio
        if self.LoadCsvFile is None:
            self.LoadCsvFile = getRamanFromFile()

        for name in sorted(os.listdir(dataroot)):
            if not os.path.isdir(os.path.join(dataroot, name)):
                continue
            if not len(os.listdir(os.path.join(dataroot, name))):
                continue
            self.name2label[name] = len(self.name2label.keys())

        self.numclasses = len(self.name2label.keys())
        self.LoadCsv(sfpath)  # 加载所有的数据文件
        # 数据分割
        self.split_data()
        try:
            self.load_raman_data()  # 数据读取
        except:  # 读取失败则重新创建文件
            self.new = True
            self.LoadCsv(sfpath)
            self.split_data()
            self.load_raman_data()

    def LoadCsv(self,
                filename, ):
        pass

    def split_data(self):
        pass

    def load_raman_data(self):
        pass

    def shuffle(self):
        z = list(zip(self.Ramans, self.labels, self.RamanFiles))
        random.shuffle(z)
        self.Ramans[:], self.labels[:], self.RamanFiles = zip(*z)
        return

    def shufflecsv(self,
                   filename = None):
        if filename is None:
            filename = self.sfpath
        path = os.path.join(self.root, filename)
        with open(path, "r") as f:
            reader = csv.reader(f)
            rows = []
            for row in reader:
                rows.append(row)
            random.shuffle(rows)
        with open(path, "w", newline = "") as f:
            writer = csv.writer(f)

            writer.writerows(rows)

        return

    def num_classes(self):
        return self.numclasses

    def file2data(self):
        return dict(zip(self.RamanFiles, self.Ramans))

    def data2file(self):
        return dict(zip(self.Ramans, self.RamanFiles))

    def name2label(self):
        return self.name2label

    def label2name(self):
        keys = list(self.name2label.values())
        values = list(self.name2label.keys())
        return dict(zip(keys, values))

    def get_data_sorted_by_label(self):
        """
        返回一个字典，每个label对应的数字指向一个tensor[label=label的光谱的数量,c=1,lenth]
        """
        spectrum_each_label = {}
        for i in range(len(self.name2label)):
            spectrum_each_label[i] = None
        for i in range(self.__len__()):
            raman, label = self.Ramans[i], self.labels[i]
            raman = torch.unsqueeze(raman, dim = 0)

            if spectrum_each_label[label.item()] is None:
                spectrum_each_label[label.item()] = raman
            else:
                spectrum_each_label[label.item()] = torch.cat(
                    (spectrum_each_label[label.item()],
                     raman),
                    dim = 0
                )
        for k in spectrum_each_label.keys():
            if spectrum_each_label[k] is None:
                continue
            if len(spectrum_each_label[k].shape) == 3:
                return spectrum_each_label
            else:
                spectrum_each_label[k] = pytorchlize(spectrum_each_label[k])
        return spectrum_each_label

    def get_data_sorted_by_sample(self):
        return self.get_data_sorted_by_label()

    def wavelengths(self):
        return torch.tensor(self.xs)

    def show_data_vis(self,
                      xs = None,
                      vis = None,
                      win = "data"):

        label2name = self.label2name()
        label2data = self.get_data_sorted_by_label()
        if xs == None:
            xs = self.wavelengths()
        if len(label2data[0].shape) > 2 and label2data[0].shape[-2] > 1:  # 多通道数据
            print("多通道数据")
            return

        if label2data[0].shape[-1] > 3:  # 光谱数据
            for i in range(self.numclasses):
                data = label2data[i]
                name = label2name[i]
                visdom_func.spectrum_vis(data, win = win + "_" + name, xs = xs, vis = vis)
        else:  # 降维后数据
            # if vis == None:
            # 	if not win.endswith(".png"):
            # 		win += ".png"
            for i in range(self.numclasses):
                data = torch.squeeze(label2data[i])
                vis.scatter(data,
                            win = win,
                            update = None if i == 0 else "append",
                            name = self.label2name()[i],
                            opts = dict(
                                title = win,
                                showlegend = True,
                            )
                            )
        return

    def get_data_sorted_by_name(self):
        name2data = {}
        label2data = self.get_data_sorted_by_label()
        for name in self.name2label.keys():
            name2data[name] = label2data[self.name2label[name]]
        return name2data

    def savedata(self,
                 dir,
                 mode = "file_wise",
                 backend = ".csv",
                 delimiter = ","):
        """

        @param dir: 保存的文件夹路径（如没有则会被创建
        @param mode: "file_wise":一类的所有光谱保存到同一个文件中；"dir_wise"：光谱按照类别保存，一个光谱一个文件
        @param backend: 文件夹后缀
        @param delimiter: 数据分隔符
        @return:
        """
        label2data = self.get_data_sorted_by_label()
        if not os.path.isdir(dir):
            os.makedirs(dir)
        for name in self.name2label.keys():
            data = label2data[self.name2label[name]]
            if mode == "file_wise":
                path = os.path.join(dir, name + backend)
                data = data.numpy()
                data = numpy.squeeze(data)
                numpy.savetxt(path, data, delimiter = delimiter)
            elif mode == "dir_wise":
                for i in range(len(data)):
                    name_dir = os.path.join(dir, name)
                    if not os.path.isdir(name_dir):
                        os.makedirs(name_dir)

                    path = os.path.join(name_dir, name + "-" + str(i) + backend)
                    # numpy.savetxt(path, data, delimiter = ",")
                    numpy.savetxt(path, numpy.vstack((self.xs, numpy.arange(len(self.xs)), data[i])).T,
                                  delimiter = delimiter,
                                  comments = "",
                                  header = "Wavelength,Column,Intensity")

            else:
                print("unsupported mode:{}".format(mode))
                raise Exception

    def __add__(self, other):
        if len(other) == 0:
            return copy.deepcopy(self)
        assert len(self.xs) == len(other.xs)
        assert self.label2name().keys() == other.label2name().keys()
        res = copy.deepcopy(self)
        for i in len(other):
            if other.RamanFiles[i] in self.RamanFiles:
                assert self.RamanFiles
            res.labels = other.labels + self.labels
            res.Ramans += other.Ramans
            res.RamanFiles += other.RamanFiles
        return res

    def __len__(self):
        return len(self.Ramans)

    def __getitem__(self,
                    item):
        assert item < len(self.Ramans), "{}/{}".format(item, len(self))

        raman, label = self.Ramans[item], self.labels[item]

        if self.unsupervised:  # label设置为noising前的光谱数据
            label = raman.to(torch.float32)
            if not self.noising == None and self.mode == "train":
                raman = torch.squeeze(raman)
                raman = self.noising(raman)
                raman = torch.unsqueeze(raman, dim = 0).to(torch.float32)

        return raman, label
