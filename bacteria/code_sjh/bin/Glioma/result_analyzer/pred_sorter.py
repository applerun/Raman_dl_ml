"""
此文件用于对未分类数据集预测结果的整理
"""
import copy
import glob
import os, pandas

result_dir = r"D:\myPrograms\pythonProject\Raman_dl_ml\bacteria\results\glioma\dl\dirwise_wjj_unlabeled"

reses = glob.glob(os.path.join(result_dir, "*", "*", "unlabeled_classfy_report.csv"))
res_df = {}


def sample_keyfunc(samplename):
    p, _, _, t = samplename.split(" ")
    return int(p[-2:] + t[1])


for res in reses:
    info = res.replace(result_dir, "")
    _, moleculer, model, _ = info.split("\\")
    df = pandas.read_csv(res)
    df.sort_values(by = "sample", inplace = True)

    pos = df["pos"].array / df["sum"].array
    if model not in list(res_df.keys()):
        res_df[model] = None
    if res_df[model] is None:
        res_df[model] = copy.deepcopy(df)
        res_df[model].drop(["neg", "pos", "sum", "true_label"], inplace = True, axis = "columns")
    res_df[model][moleculer] = pos
for key in res_df.keys():
    res_df[key].to_csv(key+".csv")
