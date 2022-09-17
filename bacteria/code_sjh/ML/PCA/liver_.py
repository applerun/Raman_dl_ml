from sklearn import svm
import os, time, csv
import numpy as np
from bacteria.code_sjh.utils.RamanData import Raman, projectroot, getRamanFromFile, Raman_dirwise
from bacteria.code_sjh.utils import Process
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
import seaborn
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from scipy import interpolate
from bacteria.code_sjh.utils.Validation.visdom_utils import startVisdomServer

startVisdomServer()


def heatmap(matrix,
            path):
    cm_fig, cm_ax = plt.subplots()
    seaborn.heatmap(matrix, annot=True, cmap="Blues", ax=cm_ax)
    cm_ax.set_title('confusion matrix')
    cm_ax.set_xlabel('predict')
    cm_ax.set_ylabel('true')
    cm_fig.savefig(path)
    plt.close(cm_fig)


def preprocess_PCA(x_train,
                   y_train,
                   x_test,
                   y_test):
    pca = PCA(n_components=2)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    return x_train, y_train, x_test, y_test


def preprocess_LDA(x_train,
                   y_train,
                   x_test,
                   y_test):
    lda = LDA(n_components=1)
    lda.fit(x_train, y_train)
    x_train = lda.transform(x_train)
    x_test = lda.transform(x_test)
    return x_train, y_train, x_test, y_test


readdatafunc0 = getRamanFromFile(wavelengthstart=390, wavelengthend=1810, )


def readdatafuncDefualt(
        filepath
):
    R, X = readdatafunc0(filepath)
    R = np.squeeze(R)
    f = interpolate.interp1d(X, R, kind="cubic")
    newX = np.linspace(400, 1800, 512)
    newR = f(newX)
    newR = np.expand_dims(newR, axis=0)

    return newR, newX


def main(
        raman=Raman_dirwise,
        k_split=6,
        readdatafunc=readdatafuncDefualt,
        transform=Process.process_series([  # 设置预处理流程
            # Process.baseline_als(),
            # Process.bg_removal_niter_fit(),
            Process.bg_removal_niter_piecewisefit(),
            Process.sg_filter(),
            Process.norm_func(), ]
        ),
        recorddir=None
):
    db_cfg = dict(
        dataroot=os.path.join(projectroot, "data", "tissue"),

        LoadCsvFile=readdatafunc,
        backEnd=".csv", t_v_t=[1., 0., 0.], newfile=False, k_split=k_split,
        transform=transform
    )
    # transform = None, )
    if recorddir is None:
        recorddir = "Record_" + time.strftime("%Y-%m-%d-%H_%M_%S")
    if not os.path.isabs(recorddir):
        recorddir = os.path.join(projectroot, "results", "tissue_ml", recorddir)  # 实验结果保存位置
    for model in [None, preprocess_PCA, preprocess_LDA]:
        recordsubdir = os.path.join(recorddir,
                                    "Record" + time.asctime().replace(":", "-").replace(" ", "_"))
        if not os.path.isdir(recordsubdir):
            os.makedirs(recordsubdir)
        recordfile = recordsubdir + ".csv"
        f = open(recordfile, "w", newline="")
        writer = csv.writer(f)
        f.write(db_cfg.__str__() + "\n")
        writer.writerow(["n", "k", "test_acc", "test_auc"])
        conf_m = None
        testaccs = []
        c = 0
        aucs = []
        for n in range(1):
            for k in range(k_split):
                sfpath = "Raman_" + str(n) + ".csv"
                train_db = raman(**db_cfg, mode="train", k=k, sfpath=sfpath)
                val_db = raman(**db_cfg, mode="val", k=k, sfpath=sfpath)
                num_classes = train_db.numclasses
                if conf_m is None:
                    conf_m = np.zeros((train_db.numclasses, train_db.numclasses))
                if not n and not k:
                    train_db.show_data(win="train")
                    val_db.show_data(win="test")
                l1 = train_db.RamanFiles
                l2 = val_db.RamanFiles
                l = list(set(l1) & set(l2))
                print(len(l) / len(val_db))
                train_data, train_label = [np.squeeze(x.numpy()) for x in train_db.Ramans], [x.item() for x in
                                                                                             train_db.labels]
                test_data, test_label = [np.squeeze(x.numpy()) for x in val_db.Ramans], [x.item() for x in
                                                                                         val_db.labels]
                if model is not None:
                    train_data, train_label, test_data, test_label = \
                        model(train_data, train_label, test_data, test_label)
                classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr',
                                     probability=True)  # ovr:一对多策略
                classifier.fit(train_data, train_label)

                pred = classifier.predict_proba(test_data)
                test_acc = classifier.score(test_data, test_label)
                testaccs.append(test_acc)

                label2auc = {}
                conf_m += confusion_matrix(test_label, classifier.predict(test_data))
                for i in range(num_classes):
                    l_t = np.equal(test_label, i).astype(int)
                    score = pred[:, i]
                    frp, tpr, thresholds = roc_curve(l_t, score)
                    plt.plot(frp, tpr)
                    # label2auc[i] = auc(frp, tpr)
                    label2auc[i] = roc_auc_score(l_t, score)
                auc_m = np.mean(list(label2auc.values()))
                aucs.append(auc_m)
                writer.writerow([n, k, test_acc, auc_m])
                c += 1
                print(c, "/", 1 * k_split)

        np.savetxt(os.path.join(recordsubdir, "test_confusion_matrix.csv"), conf_m, delimiter=",")
        heatmap(conf_m, os.path.join(recordsubdir, "test_confusion_matrix.png"))
        ta = np.mean(np.array(testaccs)).__str__() + "+-" + np.std(np.array(testaccs)).__str__()
        auca = np.mean(np.array(testaccs)).__str__() + "+-" + np.std(np.array(testaccs)).__str__()
        writer.writerow(["mean", "std", ta, auca])
        if num_classes == 2:
            A, B, C, D = conf_m.flatten()
            acc = (A + D) / (A + B + C + D)
            sens = A / (A + B)
            spec = D / (C + D)
            f.write(
                "accuracy,{}\nsensitivity,{}\nspecificity,{}".format(acc, sens, spec)
            )


def main_1(  # 留一交叉验证
        raman=Raman_dirwise,
        readdatafunc=readdatafuncDefualt,
        transform=Process.process_series([  # 设置预处理流程
            # Process.baseline_als(),
            # Process.bg_removal_niter_fit(),
            Process.bg_removal_niter_piecewisefit(),
            Process.sg_filter(),
            Process.norm_func(), ]
        ),
        recorddir=None,
        sfpath="Raman_0_filewise.csv"
):
    all_acc = []
    db_cfg = dict(
        dataroot=os.path.join(projectroot, "data", "tissue"),
        LoadCsvFile=readdatafunc,
        backEnd=".csv", t_v_t=[1., 0., 0.], newfile=False,
        transform=transform
    )
    if recorddir is None:
        recorddir = "Record_" + time.strftime("%Y-%m-%d-%H_%M_%S")
    if not os.path.isabs(recorddir):
        recorddir = os.path.join(projectroot, "results", "tissue_ml", recorddir)  # 实验结果保存位置
    for model in [None, preprocess_PCA, preprocess_LDA]:
        recordsubdir = os.path.join(recorddir,
                                    "Record" + time.asctime().replace(":", "-").replace(" ", "_"))
        if not os.path.isdir(recordsubdir):
            os.makedirs(recordsubdir)
        recordfile = recordsubdir + ".csv"
        f = open(recordfile, "w", newline="")
        writer = csv.writer(f)
        f.write(db_cfg.__str__() + "\n")
        writer.writerow(["n", "k", "test_acc", "test_auc"])

        testaccs = []
        c = 0
        # aucs = []
        loo = LeaveOneOut()
        db = raman(**db_cfg, mode="train", sfpath=sfpath)
        labels = np.array([x.detach_().cpu().numpy().__int__() for x in db.labels])
        datas = np.array([x.detach_().cpu().numpy().squeeze() for x in db.Ramans])
        files = db.RamanFiles
        num_classes = db.num_classes()
        conf_m = np.zeros((num_classes, num_classes))

        for train_idx, test_idx in loo.split(labels, datas):
            train_data, train_label = datas[train_idx], labels[train_idx]
            test_data, test_label = datas[test_idx], labels[test_idx]
            testfile = files[test_idx[0]]
            # print(len(l) / len(val_db))
            if model is not None:
                train_data, train_label, test_data, test_label = \
                    model(train_data, train_label, test_data, test_label)

            classifier = svm.SVC(
                # C = 2,
                # kernel = 'rbf', gamma = 10, decision_function_shape = 'ovr',
                probability=True
            )  # ovr:一对多策略
            classifier.fit(train_data, train_label)

            pred = classifier.predict_proba(test_data)[0]
            pred_ = int(pred[0] <= pred[1])
            # test_acc = classifier.score(test_data, test_label)

            test_ = test_label[0]
            test_acc = float(pred_ == test_)
            print(testfile, ":", test_acc)
            testaccs.append(test_acc)
            conf_m[test_, pred_] += 1
            # label2auc = {}
            # for i in range(num_classes):
            # 	l_t = np.equal(test_label, i).astype(int)
            # 	score = pred[:, i]
            # 	frp, tpr, thresholds = roc_curve(l_t, score)
            # 	plt.plot(frp, tpr)
            # label2auc[i] = auc(frp, tpr)
            # label2auc[i] = roc_auc_score(l_t, score)
            # auc_m = np.mean(list(label2auc.values()))
            # aucs.append(auc_m)
            # writer.writerow([n, k, test_acc, auc_m])
            c += 1
            print(c, "/", len(db))
            if not conf_m[0, 0] + conf_m[1, 1] == np.sum(np.array(testaccs)):
                ttt = confusion_matrix(test_label, classifier.predict(test_data))
                print("test_{}, pred_{},acc_{},pred{}".format(test_, pred_, test_acc, pred))
                del ttt
            assert conf_m.sum() == len(testaccs)
        np.savetxt(os.path.join(recordsubdir, "test_confusion_matrix.csv"), conf_m, delimiter=",")
        heatmap(conf_m, os.path.join(recordsubdir, "test_confusion_matrix.png"))
        ta = np.mean(np.array(testaccs)).__str__() + "+-" + np.std(np.array(testaccs)).__str__()
        # auca = np.mean(np.array(testaccs)).__str__() + "+-" + np.std(np.array(testaccs)).__str__()
        writer.writerow(["mean", "std", ta])
        if num_classes == 2:
            A, B, C, D = conf_m[0, 0], conf_m[0, 1], conf_m[1, 0], conf_m[1, 1]
            acc = (A + D) / (A + B + C + D)
            sens = A / (A + B)
            spec = D / (C + D)
            f.write(
                "accuracy,{}\nsensitivity,{}\nspecificity,{}".format(acc, sens, spec)
            )
            assert acc == np.mean(np.array(testaccs))


if __name__ == '__main__':
    for baselineRemoveFunc in [Process.baseline_als(),
                               Process.bg_removal_niter_fit(num_iter=30, degree=5),
                               Process.bg_removal_niter_piecewisefit(), ]:
        transform = Process.process_series([  # 设置预处理流程
            baselineRemoveFunc,
            Process.sg_filter(window_length=5, polyorder=1),
            Process.area_norm_func(), ]
        )
        main(transform = transform,raman = Raman)
        # main_1(transform=transform, raman=Raman)
