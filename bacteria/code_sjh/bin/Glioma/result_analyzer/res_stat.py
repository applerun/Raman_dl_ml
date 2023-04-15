import warnings

import pandas
import os
import numpy as np
import csv


def round_str(res: str, split = "±"):
    res = [str(round(float(x), 4)) for x in res]
    res = res[0] + split + res[1]
    return res


def RecordRead_bi_clas(src, mode = "test"):
    pd = pandas.read_csv(src + ".csv", skiprows = 2, encoding = "GBK")
    accs = pd[mode + "_acc"]
    ind = len(accs) - 2
    acc,acc_std = pd[mode + "_acc"][ind].split("+-")
    acc = round(float(acc), 4)
    acc_std = round(float(acc_std), 4)
    # acc = round_str(acc)
    auc,auc_std = pd[mode + "_AUC"][ind].split("+-")
    auc = round(float(auc), 4)
    auc_std = round(float(auc_std), 4)
    # auc = round_str(auc)
    cm = np.loadtxt(os.path.join(src, mode + "_confusion_matrix.csv"), delimiter = ",")

    (a, b), (c, d) = cm
    acc_ = (a + d) / (a + b + c + d)
    acc_ = round(acc_, 4)
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


def main_bi_class(dir, dst, mode = "test",
                  nets = None,
                  molecules = None):
    if nets is None:
        nets = ["Alexnet_Sun", "Resnet18", "Resnet34"]
    if not os.path.isabs(dst):
        dst = os.path.join(dir, dst)
    if not dst.endswith(".csv"):
        dst = dst + ".csv"
    f = open(dst, "w", newline = "")
    csv_writer = csv.writer(f)

    header1 = ["结果"] + ["Sensitivity"] * len(nets) + ["Specificity"] * len(nets) + ["Accuracy"] * len(nets) + [
        "AUC"] * len(nets)
    header = ["模型"] + nets * 4
    csv_writer.writerow(header1)
    csv_writer.writerow(header)
    if molecules is None:
        molecules = os.listdir(dir)
    for molecule in molecules:

        molecule_abs = os.path.join(dir, molecule)
        if not os.path.isdir(molecule_abs):
            continue
        row = [molecule] + [0] * 4*len(nets)
        for i in range(len(nets)):
            net = nets[i]
            src = os.path.join(molecule_abs, "Record" + net)
            res = RecordRead_bi_clas(src, mode)
            row[i + 1] = res["sen"]
            row[i + 1 + len(nets)] = res["spe"]
            row[i + 1 + len(nets) * 2] = res["acc"]
            row[i + 1 + len(nets) * 3] = res["auc"]
        csv_writer.writerow(row)
    f.close()
    return


def main(dirname = "2022-11-10-17_57_55_dirwise", nets = None):
    if nets is None:
        nets = ["Alexnet_Sun", "Resnet18", "Resnet34"]
    resuldir = os.path.join(r"D:\myPrograms\pythonProject\Raman_dl_ml\results\glioma\dl", dirname)
    dst_file = os.path.join(resuldir, "res_stat-"+dirname + ".csv")
    main_bi_class(resuldir, dst_file,nets = nets, molecules =
    "IDH(M-1)@1p19q(缺-1)@M(甲基化-1)@T(突变-1)@E(扩增-1)@7(+ 1)@10(- 1)@A(缺-1)@B(缺-1)".split("@"))


def main_2():
    """

    @return: lr wise
    """
    resultroot = r"D:\myPrograms\pythonProject\Raman_dl_ml\results\glioma\dl"

    for lr in [0.1, 0.01, 0.001]:
        src_dir = "test_base_lr{}".format(lr)
        dst_file = "res_stat_filewise_test_base_lr{}.csv".format(lr)
        main_bi_class(os.path.join(resultroot, src_dir), os.path.join(resultroot, src_dir, dst_file), mode = "test",
                      molecules =
                      "IDH(M-1)@1p-19q(缺-1)@M(甲基化-1)@T(突变-1)@E(扩增-1)@7(+ 1)@10(- 1)@A(缺-1)@B(缺-1)".split("@"))


if __name__ == '__main__':
    main("2023-04-09-21_14_47_old",["Alexnet_Sun",])
    # main_2()
