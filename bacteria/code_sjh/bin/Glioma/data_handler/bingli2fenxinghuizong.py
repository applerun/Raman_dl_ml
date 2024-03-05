import glob
import os
import shutil

import pandas
import numpy as np

molecule_short2full = {  # 《汇总结果》分子指标简写名称 与 《病理报告》 的对应关系
    "IDH(M-1)": "IDH 1",
    "1p19q(缺-1)": "1p19q",
    "M(甲基化-1)": "MGMT启动子甲基化状态",
    "T(突变-1)": "TERT（C250T）",
    "E(扩增-1)": "EGFR扩增",
    "7(+ 1)": "7号染色体",
    "10(- 1)": "10号染色体",
    "A(缺-1)": "CDKN2A纯合性缺失",
    "B(缺-1)": "CDKN2B纯合性缺失",
}

molecule_short2negpos = {  # 《汇总结果》分子指标简写中label 0，1与 《病理报告》中文字的对应关系
    "IDH(M-1)": ["W", "M"],
    "1p19q(缺-1)": ["intact", "co-deletion"],
    "M(甲基化-1)": ["非甲基化", "甲基化"],
    "T(突变-1)": ["无突变", "突变"],
    "E(扩增-1)": ["不扩增", "扩增"],
    "7(+ 1)": ["不扩增", "扩增"],
    "10(- 1)": ["不缺失", "缺失"],
    "A(缺-1)": ["不缺失", "缺失"],
    "B(缺-1)": ["不缺失", "缺失"],
}

molecule_full2short = dict(zip(list(molecule_short2full.values()), list(molecule_short2full.keys())))


def get_label(content, molecule_full):
    short = molecule_full2short[molecule_full]
    label = molecule_short2negpos[short].index(content)
    return label


def fenxing2zongjie(src, dst):
    df = pandas.read_excel(src)
    header = "	编号	组织	平均光谱	IDH(M-1)	1p19q(缺-1)	M(甲基化-1)	T(突变-1)	E(扩增-1)	7(+ 1)	10(- 1)	A(缺-1)	B(缺-1)".split(
        "\t")
    data = []
    xingmingsuoxie2xuhao = {}
    for index, row in df.iterrows():
        xuhao = row["序号"]
        xingmingsuoxie = row["姓名缩写"]
        xingmingsuoxie2xuhao[xingmingsuoxie] = xuhao
        if xuhao != xuhao:
            continue
        res = ["", xuhao, "", ""]
        next = False
        for short in header[4:]:
            full = molecule_short2full[short]
            content = row[full]
            if content != content:
                next = True
                break
            label = molecule_short2negpos[short].index(content)
            res.append(label)
        if next:
            continue
        data.append(res)

    new_df = pandas.DataFrame(columns = header, data = data)
    new_df.to_csv(dst)
    return xingmingsuoxie2xuhao


def relabel(xingmingsuoxue2xuhao, filepath):
    dir, filename = os.path.split(filepath)
    infos = filename.split(" ")
    try:
        infos[0] = xingmingsuoxue2xuhao[infos[1]]
    except:
        infos[0] = -1
    if type(infos[0]) is not str:
        infos[0] = "p"+str(int(infos[0]))

    os.rename(filepath, os.path.join(dir, " ".join(infos)))
def indexinlist(content,l:list):
    try:
        return l.index(content)
    except:
        return -1

if __name__ == '__main__':

    src = r"D:\myPrograms\pythonProject\Raman_dl_ml\data\脑胶质瘤\data_used\GBM分子分型汇总表.xlsx"
    dst = r"D:\myPrograms\pythonProject\Raman_dl_ml\data\脑胶质瘤\data_used\分子分型汇总表2.csv"
    x2x = fenxing2zongjie(src, dst)

    datapath = r"D:\myPrograms\pythonProject\Raman_dl_ml\data\脑胶质瘤\unlabeled_data\data_indep_unlabeled\unlabeled"
    for file in glob.glob(os.path.join(datapath, "*")):
        relabel(x2x,file)
    # shutil.rmtree(os.path.join("D:\myPrograms\pythonProject\Raman_dl_ml\data\脑胶质瘤","回收站"))