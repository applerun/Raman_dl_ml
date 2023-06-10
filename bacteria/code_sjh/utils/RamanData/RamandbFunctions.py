import os.path
import warnings
from bacteria.code_sjh.Core.RamanData import RamanDatasetCore
from matplotlib import pyplot as plt
def save_csv_file_info(Raman:RamanDatasetCore,dst):
    """
    TODO:根据Raman类型在dst处生成对应的信息文件
    @param Raman:
    @param dst:
    @return:
    """
    if Raman.mode is not "all":
        warnings.warn("The raman db mode is not 'all':{}, Please Check".format(Raman.mode))
    if not os.path.isdir(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    label2name = Raman.label2name()
    with open(dst,"w") as f:
        for i in range(len(Raman)):
            label = Raman[i]
            label = int(label)
            label = str(label)
            file = Raman.RamanFiles[i]






