import glob
import os.path
import shutil
import warnings
import numpy
from bacteria.code_sjh.utils.RamanData import RamanDatasetCore, Raman_dirwise, Raman, getRamanFromFile, Raman_depth_gen
from bacteria.code_sjh.utils import Process
import pandas
import torch


def get_infos(filename: str):
    """

    @param filename:
    @return:
    """
    df = pandas.read_excel(filename)
    nums = df["编号"]
    axes = df.axes[1]
    num2ele2label = {}

    for i in range(len(nums)):
        num2ele2label[nums[i]] = {}

        for j in range(4, 13):
            num = nums[i]
            labels = df[axes[j]]
            label = labels[i]
            if label != label:
                del num2ele2label[nums[i]]
                return num2ele2label
            label = int(label)
            ele = axes[j].replace("/", "-")
            ele = ele.replace("\\", "-")
            num2ele2label[num][ele] = label
    return num2ele2label


def path2func_generator(num2label):
    def path2label(file):
        file_ = os.path.basename(file)
        num = int(file_.split(" ")[0][1:])
        return num2label[num]

    return path2label


def label_RamanData(database: RamanDatasetCore, path2label_func, name2label):
    for i in range(len(database)):
        if issubclass(type(database), Raman):
            file = database.RamanFiles[i]
            file = file.split(os.sep)[1]
        elif issubclass(type(database), Raman_dirwise):
            file = database.RamanFiles[i]
            file = os.path.basename(os.path.dirname(file))
        else:
            warnings.warn("Unsupported Dataset, relabel failed")
            return
        l = path2label_func(file)
        if type(l) is not torch.Tensor:
            l = torch.tensor(l)
        database.labels[i] = l

    database.name2label = name2label
    database.numclasses = len(list(name2label.keys()))


def main(info_file: str):
    num2ele2label = get_infos(info_file)
    eles = list(num2ele2label.values().__iter__().__next__().keys())

    coderoot = "../../.."
    projectroot = "../../../.."
    data_root = os.path.join(projectroot, "data", "脑胶质瘤", "data")
    from scipy import interpolate

    readdatafunc0 = getRamanFromFile(  # 定义读取数据的函数
        wavelengthstart = 39, wavelengthend = 1810, delimeter = None,
    )

    def readdatafunc(  # 插值，将光谱长度统一为512
            filepath
    ):
        R, X = readdatafunc0(filepath)
        R = numpy.squeeze(R)
        f = interpolate.interp1d(X, R, kind = "cubic")
        newX = numpy.linspace(400, 1800, 512)
        newR = f(newX)
        newR = numpy.expand_dims(newR, axis = 0)
        return newR, newX

    db_cfg = dict(  # 数据集设置
        dataroot = data_root,
        backEnd = ".csv",
        # backEnd = ".asc",
        t_v_t = [0.8, 0.2, 0.0],
        LoadCsvFile = readdatafunc,
        k_split = 6,
        transform = Process.process_series([  # 设置预处理流程
            # Process.baseline_als(),
            # Process.bg_removal_niter_fit(),
            Process.bg_removal_niter_piecewisefit(),
            Process.sg_filter(),
            Process.norm_func(), ]
        )
    )
    raman = Raman_depth_gen(2, 2)
    for ele in eles:
        num2label = {}
        for k in num2ele2label.keys():
            num2label[k] = num2ele2label[k][ele]
        name2label = {"0": 0, "1": 1}
        label2name = {"0": "neg", "1": "pos"}
        db = raman(**db_cfg, sfpath = "Raman_{}_unlabeled.csv".format(ele), newfile = True, shuffle = False)
        path2labelfunc = path2func_generator(num2label)
        label_RamanData(db, path2labelfunc, name2label)
        new_tree = os.path.join(os.path.dirname(data_root),"data_new",ele)
        reform_tree(data_root,new_tree,path2labelfunc)
        for dir in os.listdir(new_tree):
            dir_abs = os.path.join(new_tree,dir)
            new_dir = label2name[dir]
            os.rename(dir_abs,os.path.join(new_tree,new_dir))

    return

def reform_tree(src_root,dst_root,path2labelfunc):
    for dir in glob.glob(os.path.join(src_root,"*","*")):
        label = path2labelfunc(dir)
        dst_dir = os.path.join(dst_root,str(label))
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        shutil.copytree(dir,os.path.join(dst_dir,os.path.basename(dir)))





if __name__ == '__main__':
    main(r"D:\myPrograms\pythonProject\Raman_dl_ml\bacteria\data\脑胶质瘤\data_used\第一二三批 病例编号&大类结果.xlsx")