import os.path
import warnings
import numpy
from bacteria.code_sjh.utils.RamanData import RamanDatasetCore, Raman_dirwise, Raman, getRamanFromFile, Raman_depth_gen
from bacteria.code_sjh.utils import Process
import pandas


def get_infos(filename: str):
    df = pandas.read_excel(filename)
    nums = df["编号"]
    axes = df.axes[1]
    num2ele2label = {}

    for i in range(len(nums)):
        num2ele2label[nums[i]] = {}
        for j in range(4, 13):
            num2ele2label[nums[i]][axes[j]] = int(df[axes[j]][i])
    return num2ele2label


def path2func_generator(num2label):
    def path2label(file):
        file_ = os.path.basename(file)
        num = int(file_.split(" ")[0][1:])
        return num2label[num]

    return path2label


def label_RamanData(database: RamanDatasetCore, path2label_func, name2label):
    for i in range(len(database)):
        if issubclass(type(database),Raman):
            file = database.RamanFiles[i]
            file = file.split(os.sep)[1]
        elif issubclass(type(database),Raman_dirwise):
            file = database.RamanFiles[i]
            file = os.path.basename(os.path.dirname(file))
        else:
            warnings.warn("Unsupported Dataset, relabel failed")
            return
        database.labels[i] = path2label_func(file)
    database.name2label = name2label


def main(info_file: str):
    num2ele2label = get_infos(info_file)
    eles = list(num2ele2label.values().__iter__().__next__().keys())

    coderoot = "../../.."
    projectroot = "../../../.."
    data_root = os.path.join(projectroot, "data", "脑胶质瘤", "data")
    from scipy import interpolate

    readdatafunc0 = getRamanFromFile(  # 定义读取数据的函数
        wavelengthstart = 39, wavelengthend = 1810, delimeter = None,
        # dataname2idx = {"Wavelength": 0, "Intensity": 1}
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
    raman = Raman_depth(2,2)
    for ele in eles:
        num2label = {}
        for k in num2ele2label.keys():
            num2label[k] = num2ele2label[k][ele]
        name2label = {"0": 0, "1": 1}
        db = raman(**db_cfg, sfpath = "Raman_{}_unlabeled.csv".format(ele), newfile = True, max_depth = 2)
        path2labelfunc = path2func_generator(num2label)
        label_RamanData(db, path2labelfunc, name2label)
    return


if __name__ == '__main__':
    main(r"D:\myPrograms\pythonProject\Raman_dl_ml\bacteria\data\脑胶质瘤\data_used\第一二三批 病例编号&大类结果.xlsx")
