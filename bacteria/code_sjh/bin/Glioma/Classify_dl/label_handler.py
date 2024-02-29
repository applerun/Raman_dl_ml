import glob
import os.path
import shutil
import warnings
import numpy
from bacteria.code_sjh.utils.RamanData import RamanDatasetCore, Raman_dirwise, Raman, Raman_depth_gen
from bacteria.code_sjh.Core.basic_functions.fileReader import getRamanFromFile
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

        for j in range(4, 11):
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


def path2func_generator(num2label, prefix_len = 0, delimeter = " ", index = 0, func = None):
    """
    @param num2label: 序号与标签的对应关系
    @param prefix_len: 序号prefix，例如文件名为 p02 XXX ****** T1 则设置改变量为1（忽略字母p）
    @param index: 序号在文件名所包含的信息第index个信息中，0代表第一个信息，例如文件名为 XXX p02 ****** T1 时将该变量设置为 1
    @param delimeter: 文件名包含信息的分割符号，例如文件名为 XXX_p02_******_T1 时将该变量设置为 "_"
    @param func: 对数字的额外操作，例如希望将编号01记录为1001时，func=lambda x:1000+x
    @return: None
    """

    def path2label(file):
        """

        @param file: 文件名称
        """
        file_ = os.path.basename(file)
        num = int(file_.split(delimeter)[index][prefix_len:])
        if func is not None:
            num = func(num)
        try:
            return num2label[num]
        except:
            return -1

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


def main(info_file: str, src = "data_indep_unlabeled", dst = "data_indep"):
    num2ele2label = get_infos(info_file)
    eles = list(num2ele2label.values().__iter__().__next__().keys())

    projectroot = getRootPath("Raman_dl_ml")
    data_root = os.path.join(projectroot, "data", "脑胶质瘤", src)
    from scipy import interpolate

    readdatafunc = getRamanFromFile(  # 定义读取数据的函数
        wavelengthstart = 39, wavelengthend = 1810, delimeter = None,
    )

    db_cfg = dict(  # 数据集设置
        dataroot = data_root,
        backEnd = ".csv",
        # backEnd = ".asc",
        mode = "all",
        LoadCsvFile = readdatafunc,
        transform = None,
        # Process.process_series([  # 设置预处理流程
        #     Process.interpolator(),
        #     # Process.baseline_als(),
        #     # Process.bg_removal_niter_fit(),
        #     Process.bg_removal_niter_piecewisefit(),
        #     Process.sg_filter(),
        #     Process.norm_func(), ]
        # )
    )
    raman = Raman_depth_gen(2, 2)

    for ele in eles:
        num2label = {}

        for k in num2ele2label.keys():
            num2label[k] = num2ele2label[k][ele]
        path2labelfunc = path2func_generator(num2label,prefix_len = 0 if "indep" in src else 1, func = (lambda x: x + 1000) if "indep" in src else None)
        name2label = {"neg": 0, "pos": 1}
        label2name = {"0": "neg", "1": "pos","-1":"unknown"}
        db = raman(**db_cfg, sfpath = "Raman_{}_unlabeled.csv".format(ele), newfile = True, shuffle = False)

        label_RamanData(db, path2labelfunc, name2label)
        new_tree = os.path.join(os.path.dirname(data_root), dst, ele)
        if os.path.isdir(new_tree):
            shutil.rmtree(new_tree)
        reform_tree(data_root, new_tree, path2labelfunc)
        for dir in os.listdir(new_tree):
            dir_abs = os.path.join(new_tree, dir)
            new_dir = label2name[dir]
            os.rename(dir_abs, os.path.join(new_tree, new_dir))

    return


def reform_tree(src_root, dst_root, path2labelfunc):
    for dir in glob.glob(os.path.join(src_root, "*", "*")):
        try:
            label = path2labelfunc(dir)
        except:
            continue
        dst_dir = os.path.join(dst_root, str(label))
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        shutil.copytree(dir, os.path.join(dst_dir, os.path.basename(dir)))
        print("label:",label,"; dir: ",os.path.basename(dir)," done")


if __name__ == '__main__':
    from bacteria.code_sjh.Core.basic_functions.path_func import getRootPath
    datapath = os.path.join(getRootPath("Raman_dl_ml"),"data\脑胶质瘤\data_used\病例编号&分类结果2.xlsx")
    main(datapath,src = "data",dst = "data_batch123")
