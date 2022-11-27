import warnings

import pandas
import os
import numpy as np
import csv


def round_str(res: str, split = "±"):
    res = [str(round(float(x), 4)) for x in res]
    res = res[0] + split + res[1]
    return res


def RecordRead_bi_clas(src):
    pd = pandas.read_csv(src + ".csv", skiprows = 2, encoding = "GBK")
    accs = pd["test_acc"]
    ind = len(accs) - 2
    acc = pd["test_acc"][ind].split("+-")[0]
    acc = round(float(acc),4)
    # acc = round_str(acc)
    auc = pd["test_AUC"][ind].split("+-")[0]
    auc = round(float(auc), 2)
    # auc = round_str(auc)
    cm = np.loadtxt(os.path.join(src, "test_confusion_matrix.csv"), delimiter = ",")

    (a, b), (c, d) = cm
    acc_ = (a + d) / (a + b + c + d)
    acc_ = round(acc_,4)
    sen = d / (c + d)
    sen = round(sen, 4)
    spe = a / (a + b)
    spe = round(spe, 4)
    # acc_f = float(acc.split("±")[0])
    acc_f = acc
    if abs(acc_ - acc_f) > 0.01:
        warnings.warn("acc from conf_m({}) and recordfile({}) are different".format(acc_, acc))

    return dict(acc = acc_, auc = auc, sen = sen, spe = spe)


def RecordRead_Polytomous(src, class_of_interest = -1):
    return


def main_bi_class(dir, dst):
    f = open(dst, "w", newline = "")
    csv_writer = csv.writer(f)
    nets = ["Alexnet_Sun", "Resnet18", "Resnet34"]
    header1 = ["结果"] + ["Sensitivity"] * 3 + ["Specificity"] * 3 + ["Accuracy"] * 3 + ["AUC"] * 3
    header = ["模型"] + ["Alexnet", "Resnet18", "Resnet34"] * 4
    csv_writer.writerow(header1)
    csv_writer.writerow(header)
    for molecule in os.listdir(dir):
        molecule_abs = os.path.join(dir, molecule)
        row = [molecule] + [0] * 12
        for i in range(len(nets)):
            net = nets[i]
            src = os.path.join(molecule_abs, "Record" + net)
            res = RecordRead_bi_clas(src)
            row[i + 1] = res["sen"]
            row[i + 4] = res["spe"]
            row[i + 7] = res["acc"]
            row[i + 10] = res["auc"]
        csv_writer.writerow(row)
    f.close()
    return


if __name__ == '__main__':
    resultroot = r"D:\myPrograms\pythonProject\Raman_dl_ml\bacteria\results\glioma\dl"
    for lr in [0.1,0.01,0.001]:
        src_dir = "test_base_lr{}".format(lr)
        dst_file = "res_stat_filewise_test_base_lr{}.csv".format(lr)

        main_bi_class(os.path.join(resultroot,src_dir),dst_file)

