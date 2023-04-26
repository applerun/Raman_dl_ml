import warnings

import csv, seaborn
from torch import optim

from bacteria.code_sjh.models.CNN.AlexNet import AlexNet_Sun
from bacteria.code_sjh.models.CNN.ResNet import ResNet18, ResNet34
from bacteria.code_sjh.utils.Validation.validation import *
from bacteria.code_sjh.utils.RamanData import getRamanFromFile
from bacteria.code_sjh.Core.basic_functions.visdom_func import *
from bacteria.code_sjh.Core.basic_functions.mpl_func import *
from bacteria.code_sjh.utils.iterator import train
from label_handler import get_infos, path2func_generator
from torch.utils.data import DataLoader

global loss
import pysnooper
from bacteria.code_sjh.utils import Process

torch.backends.cudnn.benchmark = True

coderoot = "../../.."
projectroot = "../../../../../.."
logfile = os.path.join(projectroot, "log", "glioma", "DL_classification", os.path.basename(__file__),
                       time.strftime("%Y-%m-%d-%H_%M_%S") + ".txt")
if not os.path.isdir(os.path.dirname(logfile)):
    os.makedirs(os.path.dirname(logfile))
with open(logfile, "w") as f:
    pass

startVisdomServer()  # python -m visdom.server    启动visdom本地服务器
vis = visdom.Visdom()  # visdom对象
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运算设备


@pysnooper.snoop(
    logfile,
    prefix = "--*--")
def train_classification_net(
        net,
        train_db,
        val_db,
        test_db,
        device,
        # 选择训练键
        lr = 0.002,
        # 选择learning rate
        epochs = 100,
        # 选择epoch
        batchsz = 40,
        # 选择batch size
        numworkers = 0,
        modelname = None,
        # 选择保存名
        vis = None,
        save_dir = os.path.join(coderoot, "checkpoints", ),  # 选择保存目录

        # 每隔epo_interv 验证一次
        # prog_bar = False,  # 实现进度条功能
        criteon = nn.CrossEntropyLoss(),
        # 选择criteon
        verbose = False,
        epo_interv = 2,
        k = 0.,
        show_test_acc_line = False,
):
    """

    @param net:  训练的网络
    @param train_db:  训练集
    @param val_db: 验证集
    @param test_db: 测试集
    @param device:  训练设备
    @param lr:  学习率
    @param epochs:  最大迭代次数
    @param batchsz: batch size
    @param numworkers: data loader 线程数，建议默认为0
    @param modelname: 模型名称，None则采用默认名称，影响保存的文件名
    @param vis: visdom可视化对象
    @param save_dir: 保存文件夹
    @param criteon: 损失函数
    @param verbose: 是否冗余报告（每隔epo_interv 验证一次性能)
    @param epo_interv: 每隔epo_interv 测试一次性能（需要verbose 设置为True）
    @param k: 多折交叉验证的第k+1折（默认为0）
    @return:    res = dict(train_acces = train_acces, val_acces = val_acces,  # 训练过程——正确率
               train_losses = train_losses, val_losses = val_losses,  # 训练过程——损失函数
               best_acc = best_acc, best_epoch = best_epoch,  # early-stopping位置
               res_test = res_test,  # 测试集所有指标：
               # acc：float正确率, loss:float,
               # label2roc:dict 各个label的ROC, label2auc:dict 各个label的AUC, confusion_matrix:np.ndarray 混淆矩阵
               res_val = res_val,  # 验证集所有指标
               )
    """
    if vis == None:
        vis = visdom.Visdom()

    if not modelname is None:
        net.set_model_name(modelname)

    optimizer = optim.Adam(net.parameters(), lr = lr)

    # create loaders\
    train_loader = DataLoader(
        train_db, batch_size = batchsz, shuffle = True, num_workers = numworkers,
    )
    val_loader = DataLoader(val_db, batch_size = batchsz, num_workers = numworkers)
    test_loader = DataLoader(test_db, batch_size = batchsz, num_workers = numworkers)
    print("data loaded")
    # optimizer
    best_acc, best_epoch = 0, 0
    best_loss = 0
    global_step = 0

    steps = []
    train_losses, val_losses, test_losses = [], [], []
    train_acces, val_acces, test_accses = [], [], []

    vis.line([0], [-1], win = "loss_" + str(k), opts = dict(title = "loss_" + str(k)))
    vis.line([0], [-1], win = "val_acc_" + str(k), opts = dict(title = "val_acc_" + str(k)))
    vis.line([0], [-1], win = "train_acc_" + str(k), opts = dict(title = "train_acc_" + str(k), name = "train_acc"))
    if show_test_acc_line:
        vis.line([0], [-1], win = "val_acc_" + str(k), opts = dict(title = "val_acc_" + str(k), ), name = "test_acc")
    save_dir = os.path.join(save_dir, net.model_name + ".mdl")

    if not os.path.exists(save_dir):
        with open(save_dir, "w", newline = ""):
            pass

    print("start training")
    for epoch in range(epochs):

        loss = train(net, lr, device, train_loader, criteon, optimizer, epoch, mixed = False)
        vis.line([loss.item()], [global_step], win = "loss_" + str(k), update = "append")
        global_step += 1

        net.eval()
        with torch.no_grad():
            val_loss = evaluate_loss(net, val_loader, criteon, device)
            # test_loss = evaluate_loss(net,test_loader,criteon,device)
            train_acc = evaluate(net, train_loader, device)
            val_acc = evaluate(net, val_loader, device)
            # test_acc = evaluate(net, test_loader, device)

            train_acces.append(train_acc)
            val_acces.append(val_acc)
            # test_accses.append(test_acc)
            train_losses.append(loss)
            val_losses.append(val_loss)
            # test_losses.append(test_loss)
            steps.append(global_step)

            vis.line([val_acc], [global_step], win = "val_acc_" + str(k), update = "append")
            vis.line([train_acc], [global_step], win = "train_acc_" + str(k), update = "append")
            # vis.line([test_acc], [global_step], win = "test_acc_" + str(k), update = "append")
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                net.save(save_dir)

        if epoch % epo_interv == 0 or epoch == epochs - 1:
            if verbose:
                sample2acc_train = evaluate_samplewise(net, val_db, device)
                sample2acc_val = evaluate_samplewise(net, val_db, device)
                sample2acc_test = evaluate_samplewise(net, test_db, device)
                batch_plt(
                    sample2acc_val, global_step, win = "val_acc_each_sample" + str(k),
                    update = None if global_step <= epo_interv else "append", viz = vis
                )
                batch_plt(
                    sample2acc_test, global_step, win = "test_acc_each_sample" + str(k),
                    update = None if global_step <= epo_interv else "append", viz = vis
                )
                batch_plt(sample2acc_train, global_step, win = "val_acc_each_sample" + str(k),
                          update = None if global_step <= epo_interv else "append", viz = vis)

    net.load(save_dir)

    print("best_acc:", best_acc, "best epoch", best_epoch)
    # print("loaded from ckpt!")
    res_test = evaluate_all(net, test_loader, criteon, device)
    res_val = evaluate_all(net, val_loader, criteon, device)
    res = dict(train_acces = train_acces, val_acces = val_acces,  # 训练过程——正确率
               train_losses = train_losses, val_losses = val_losses,  # 训练过程——损失函数
               best_acc = best_acc, best_epoch = best_epoch,  # early-stopping位置
               res_test = res_test,  # 测试集所有指标：
               # acc：float正确率, loss:float,
               # label2roc:dict 各个label的ROC, label2auc:dict 各个label的AUC, confusion_matrix:np.ndarray 混淆矩阵
               res_val = res_val,  # 验证集所有指标

               )

    label2data = val_db.get_data_sorted_by_label()
    cams = {}
    for l in label2data.keys():
        data = label2data[l]
        if data is None:
            continue
        data = data.to(device)
        test_cam = grad_cam(net, data, l, device)
        cams[l] = test_cam
    res["val_cam"] = cams
    label2data = test_db.get_data_sorted_by_label()
    cams = {}
    for l in label2data.keys():
        data = label2data[l]
        if data is None:
            cams[l] = None
            continue
        data = data.to(device)
        test_cam = grad_cam(net, data, l, device)
        cams[l] = test_cam
    res["test_cam"] = cams

    # sample2acc_test = evaluate_samplewise(alexnet, val_db + test_db, device)
    # print(sample2acc_test)
    return res


def plt_loss_acc(
        pltdir,
        res,
        informations = None):
    trainfig, trainax = plt.subplots(1, 2)  # 绘制训练过程图
    trainfig.suptitle('train process' + informations)

    epochs = np.arange(len(res["train_losses"]))

    loss_ax, acc_ax = trainax

    loss_ax.set_title("loss")
    loss_ax.plot(epochs, res["train_losses"], label = "train_loss", color = "red")
    loss_ax.plot(epochs, res["val_losses"], label = "val_loss", color = "blue")
    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("loss")
    loss_ax.legend()

    acc_ax.set_title("accuracy")
    acc_ax.plot(epochs, res["train_acces"], label = "train_accuracy", color = "red")
    acc_ax.plot(epochs, res["val_acces"], label = "val_accuracy", color = "blue")
    acc_ax.set_xlabel("epoch")
    acc_ax.set_ylabel("accuracy")
    acc_ax.legend()
    plt.subplots_adjust(wspace = 0.25)
    trainfig.savefig(os.path.join(pltdir, "train_process.png"))
    plt.close(trainfig)


def plt_res_val(pltdir,
                res,
                label2name,
                informations = None):
    if informations is None:
        informations = ""
    label2roc = res["label2roc"]
    label2auc = res["label2auc"]
    for label in label2roc.keys():
        fpr, tpr, thresholds = label2roc[label]
        auc = label2auc[label]
        roc_fig, roc_ax = plt.subplots()
        roc_fig.suptitle("ROC_curve")
        roc_ax.set_title("auc = {}".format(auc))
        roc_ax.plot(fpr, tpr)
        roc_ax.set_xlabel("fpr")
        roc_ax.set_ylabel("tpr")
        roc_fig.savefig(os.path.join(pltdir, informations + "_" + label2name[label] + "_roc.png"))
        plt.close(roc_fig)
    confusion_matrix = res["confusion_matrix"]
    cm_fig, cm_ax = plt.subplots()
    seaborn.heatmap(confusion_matrix, annot = True, cmap = "Blues", ax = cm_ax)
    cm_ax.set_title('confusion matrix')
    cm_ax.set_xlabel('predict')
    cm_ax.set_ylabel('true')
    cm_fig.savefig(os.path.join(pltdir, informations + "_confusion_matrix.png"))
    plt.close(cm_fig)


# def replt_cam(pltdir,db: RamanDatasetCore):
# 	camdir = os.path.join(pltdir,"cam")
# 	name2label = db.name2label()
# 	label2data = db.get_data_sorted_by_label()
# 	for file in os.listdir(camdir):
# 		if not file.endswith(".csv"):
# 			continue
# 		file_p = os.path.join(camdir,file)
# 		file_n = file[:-4]
# 		information,_,name = file.split("_")[0:3]
# 		label =name2label[name]

def plt_cam(pltdir,
            cams,
            db: RamanDatasetCore,
            informations = None):
    if informations is None:
        informations = ""
    label2name = db.label2name()
    label2data = db.get_data_sorted_by_label()
    fig, ax = plt.subplots(db.numclasses, 1)
    fig.suptitle(informations + "_grad_cam")
    for label in cams.keys():
        cam_ax = ax[label]
        cam = cams[label]
        if cam is None:
            continue
        name = label2name[label]
        data = label2data[label]
        spectrum_vis_mpl(data, xs = db.xs, name = 'spectrum_' + name, ax = cam_ax, line_color = "blue",
                         shadow_color = "skyblue")
        spectrum_vis_mpl(cam, xs = np.linspace(db.xs[0], db.xs[-1], cam.shape[-1]), name = "cam_" + name, ax = cam_ax,
                         line_color = "red", shadow_color = "pink")
        cam_ax.legend()
    plt.subplots_adjust(wspace = 0.25)
    fig.savefig(os.path.join(pltdir, informations + "_grad_cam"))
    plt.close(fig)


def plt_res(pltdir,
            res,
            val_db,
            test_db,
            informations = None):
    if informations is None:
        informations = ""
    label2name = val_db.label2name()

    plt_loss_acc(pltdir, res, informations)
    plt_res_val(pltdir, res["res_val"], label2name, informations = "val")
    plt_res_val(pltdir, res["res_test"], label2name, informations = "test")
    plt_cam(pltdir, res["val_cam"], val_db, "val")
    plt_cam(pltdir, res["test_cam"], test_db, "val")


def heatmap(matrix,
            path):
    cm_fig, cm_ax = plt.subplots()
    seaborn.heatmap(matrix, annot = True, cmap = "Blues", ax = cm_ax)
    cm_ax.set_title('confusion matrix')
    cm_ax.set_xlabel('predict')
    cm_ax.set_ylabel('true')
    cm_fig.savefig(path)
    plt.close(cm_fig)


def npsv(pltdir,
         res,
         val_db,
         test_db,
         ):
    process = np.array([res["train_acces"], res["val_acces"], res["train_losses"], res["val_losses"]]).T
    np.savetxt(os.path.join(pltdir, "train_process.csv"), process,
               header = "train_acces,val_acces,train_losses,val_losses", delimiter = ",")
    os.makedirs(os.path.join(pltdir, "cam"))
    for label in test_db.label2name().keys():
        name = test_db.label2name()[label]
        try:
            val_cam = res["val_cam"][label]
            test_cam = res["test_cam"][label]
            xs = np.linspace(val_db.xs[0], val_db.xs[-1], val_cam.shape[-1])
            xs = np.expand_dims(xs, axis = 0)

            np.savetxt(os.path.join(pltdir, "cam", "val_cam_" + name + "_activated.csv"), np.vstack((xs, val_cam)),
                       delimiter = ",")
            np.savetxt(os.path.join(pltdir, "cam", "test_cam_" + name + "_activated.csv"), np.vstack((xs, test_cam)),
                       delimiter = ",")
        except:
            warnings.warn("cam output failed")
    return


readdatafunc = getRamanFromFile(  # 定义读取数据的函数
    wavelengthstart = 39, wavelengthend = 1810, delimeter = None,
    dataname2idx = {"Wavelength": 2, "Intensity": 6}
)


def train_modellist(
        dataroot,
        db_cfg = None,
        raman = RamanDatasetCore,
        # 设置读取数据集的DataSet
        # 设置k叠交叉验证的k值
        modellist = None,
        recorddir = None,
        path2labelfunc = None,
        test_db = None,
        sfname = "Raman_",
        n_iter = 1  # 交叉验证重复次数
):
    if recorddir is None:
        recorddir = "Record_" + time.strftime("%Y-%m-%d-%H_%M_%S")
        recorddir = os.path.join(projectroot, "results", "tissue_dl", recorddir)
    if modellist is None:
        modellist = [AlexNet_Sun, ResNet18, ResNet34]
    if db_cfg is None:
        db_cfg = dict(  # 数据集设置
            dataroot = dataroot,
            backEnd = ".csv",
            # backEnd = ".asc",
            t_v_t = [0.8, 0.2, 0.0],
            LoadCsvFile = readdatafunc,
            k_split = 6,
            transform = Process.process_series([  # 设置预处理流程
                Process.intorpolator(),
                # Process.baseline_als(),
                Process.bg_removal_niter_fit(),
                # Process.bg_removal_niter_piecewisefit(),
                Process.sg_filter(),
                Process.norm_func(), ]
            )
        )
    k_split = db_cfg["k_split"]
    bestaccs, testaccs, bepochs, vaucs, taucs = [], [], [], [], []

    # readdatafunc = getRamanFromFile(wavelengthstart = 400,wavelengthend = 1800,delimeter = delimeter,dataname2idx = dataformat)

    train_cfg = dict(  # 训练参数
        device = device,
        batchsz = 100,
        vis = vis,
        lr = 0.001,
        epochs = 300,
        verbose = False,
    )
    # config = dict(dataroot = os.path.join(projectroot, "data", "data_AST"), backEnd = backend, t_v_t = tvt, LoadCsvFile = getRamanFromFile(wavelengthstart = 0, wavelengthend = 1800, delimeter = delimeter,
    #                                  dataname2idx = dataformat), k_split = k_split)


    i = 0  # 实验进度计数

    # recorddir = os.path.join(projectroot, "results", "liver", recorddir)  # 实验结果保存位置

    if not os.path.isdir(recorddir):
        os.makedirs(recorddir)
    for model in modellist:
        recordsubdir = os.path.join(recorddir,
                                    "Record" + model.__name__)  # + time.asctime().replace(":", "-").replace(" ", "_"))  # 每个模型一个文件夹保存结果
        if not os.path.isdir(recordsubdir):
            os.makedirs(recordsubdir)
        recordfile = recordsubdir + ".csv"  # 记录训练的配置和结果

        f = open(recordfile, "w", newline = "")
        writer = csv.writer(f)
        f.write(db_cfg.__str__() + "\n")
        f.write(train_cfg.__str__() + "\n")
        writer.writerow(["n", "k", "best_acc", "test_acc", "best_epoch", "val_AUC", "test_AUC"])
        conf_m_v = None
        conf_m_t = None
        for n in range(n_iter):
            for k in range(k_split):
                sfpath = sfname + str(n) + ".csv"

                train_db = raman(**db_cfg, mode = "train", k = k, sfpath = sfpath, newfile = True if n > 0 else False)
                val_db = raman(**db_cfg, mode = "val", k = k, sfpath = sfpath)
                if db_cfg["t_v_t"][2] == 0 and test_db is None:
                    test_db = val_db
                elif test_db is None:
                    test_db = raman(**db_cfg, mode = "test", k = k, sfpath = sfpath)

                train_db.show_data_vis()
                # __________________label_______________________
                # label_RamanData(train_db, path2labelfunc, name2label)
                # label_RamanData(val_db, path2labelfunc, name2label)
                if conf_m_v is None:
                    conf_m_v = np.zeros((train_db.numclasses, train_db.numclasses))
                    conf_m_t = np.zeros((train_db.numclasses, train_db.numclasses))
                assert len(val_db) > 0, str(val_db.sfpath) + ":" + str(val_db.RamanFiles)
                # test_db = raman(**db_cfg, mode = "test", k = k, sfpath = sfpath)
                # train_db.show_data()
                sample_tensor, sample_label = train_db.__getitem__(0)
                vis.line(sample_tensor, win = "sampletensor")

                sample_tensor = torch.unsqueeze(sample_tensor, dim = 0)

                net = model(sample_tensor, train_db.num_classes()).to(device)
                res = train_classification_net(
                    net,
                    train_db =train_db,
                    val_db = val_db,
                    test_db = test_db,
                    **train_cfg,
                    k = k
                )

                pltdir = os.path.join(recordsubdir, "n-{}-k-{}".format(n, k))
                if not os.path.isdir(pltdir):
                    os.makedirs(pltdir)
                b, t, be, auc_val, auc_test = res["best_acc"], res["res_test"]["acc"], res["best_epoch"], \
                                              np.mean(list(res["res_val"]["label2auc"].values())), \
                                              np.mean(list(res["res_test"]["label2auc"].values()))
                assert b == res["res_val"]["acc"]
                writer.writerow([n, k, b, t, be, auc_val, auc_test])
                bestaccs.append(b)
                testaccs.append(t)
                bepochs.append(be)
                vaucs.append(auc_val)
                taucs.append(auc_test)
                i += 1
                print(i, "/", len(modellist) * n_iter * k_split)
                plt_res(pltdir,
                        res,
                        val_db,
                        test_db,
                        informations = None)
                npsv(pltdir, res, val_db, val_db, )
                shape = res["res_val"]["confusion_matrix"].shape
                assert shape[0] == shape[1] == train_db.numclasses
                shape = res["res_test"]["confusion_matrix"].shape
                assert shape[0] == shape[1] == train_db.numclasses
                conf_m_v += res["res_val"]["confusion_matrix"]
                conf_m_t += res["res_test"]["confusion_matrix"]
        np.savetxt(os.path.join(recordsubdir, "val_confusion_matrix.csv"), conf_m_v, delimiter = ",")
        np.savetxt(os.path.join(recordsubdir, "test_confusion_matrix.csv"), conf_m_t, delimiter = ",")

        heatmap(conf_m_t, os.path.join(recordsubdir, "test_confusion_matrix.png"))
        heatmap(conf_m_v, os.path.join(recordsubdir, "val_confusion_matrix.png"))
        # train_db.shufflecsv()
        ba = np.mean(np.array(bestaccs)).__str__() + "+-" + np.std(np.array(bestaccs)).__str__()
        ta = np.mean(np.array(testaccs)).__str__() + "+-" + np.std(np.array(testaccs)).__str__()
        bea = np.mean(np.array(bepochs)).__str__() + "+-" + np.std(np.array(bepochs)).__str__()
        auc_av = np.mean(np.array(vaucs)).__str__() + "+-" + np.std(np.array(vaucs)).__str__()
        auc_at = np.mean(np.array(taucs)).__str__() + "+-" + np.std(np.array(taucs)).__str__()

        writer.writerow(["mean", "std", ba, ta, bea, auc_av, auc_at])
        f.write("\n" + net.model_name)
        f.close()
        print("best acc:", ba)
        print("test acc", ta)
        print("best epochs", bea)


def main_indep_test():
    from bacteria.code_sjh.utils import Process

    info_file = r"D:\myPrograms\pythonProject\Raman_dl_ml\bacteria\data\脑胶质瘤\data_used\病例编号&分类结果2.xlsx"
    num2ele2label = get_infos(info_file)
    eles = list(num2ele2label.values().__iter__().__next__().keys())

    print("using device:", device.__str__())
    dataroot_ = os.path.join(projectroot, "data", "脑胶质瘤", "data_classified")
    dataroot_test_ = os.path.join(projectroot, "data", "脑胶质瘤", "data_indep")
    recordroot = os.path.join(projectroot, "results", "glioma", "dl")
    # raman = Raman_depth_gen(2, 2)  # TODO:根据数据存储方式选择合适的读取策略（Raman/Raman_dirwise)
    raman = Raman_dirwise
    recordroot = os.path.join(recordroot, time.strftime("%Y-%m-%d-%H_%M_%S"))

    paras = [(e, pre) for e in eles for pre in [
        # Process.baseline_als(),
        Process.bg_removal_niter_fit(),
        # Process.bg_removal_niter_piecewisefit(),
    ]]

    for ele, preprocess in paras:
        num2label = {}
        for k in num2ele2label.keys():
            num2label[k] = num2ele2label[k][ele]
        name2label = {"neg": 0, "pos": 1}
        dataroot = os.path.join(dataroot_, ele)
        dataroot_test = os.path.join(dataroot_test_, ele)
        modellist = [AlexNet_Sun, ResNet18, ResNet34]
        db_cfg = dict(  # 数据集设置
            dataroot = dataroot_test,
            backEnd = ".csv",
            # backEnd = ".asc",
            t_v_t = [0.8, 0.2, 0.0],
            LoadCsvFile = readdatafunc,
            k_split = 6,
            transform = Process.process_series([  # 设置预处理流程
                Process.intorpolator(),
                Process.sg_filter(),
                preprocess,
                Process.norm_func(), ]
            ))
        test_db = Raman_dirwise(**db_cfg)
        db_cfg["dataroot"] = dataroot
        recorddir = os.path.join(recordroot, ele)
        path2labelfunc = path2func_generator(num2label)
        train_modellist(dataroot, db_cfg = db_cfg, raman = raman, modellist = modellist, recorddir = recorddir,
                        path2labelfunc = path2labelfunc, test_db = test_db)


# 设置数据集分割
def main_one_datasrc(
        dataroot_ = os.path.join(projectroot, "data", "脑胶质瘤", "data_all")
):
    info_file = r"D:\myPrograms\pythonProject\Raman_dl_ml\data\脑胶质瘤\data_used\病例编号&分类结果2.xlsx"
    num2ele2label = get_infos(info_file)
    eles = list(num2ele2label.values().__iter__().__next__().keys())
    print("using device:", device.__str__())
    recordroot = os.path.join(projectroot, "results", "glioma", "dl")
    # raman = Raman_depth_gen(2, 2)  # TODO:根据数据存储方式选择合适的读取策略（Raman/Raman_dirwise)
    raman = Raman_dirwise
    recordroot = os.path.join(recordroot, time.strftime("%Y-%m-%d-%H_%M_%S"))

    paras = [(e, pre) for e in eles for pre in [
        # Process.baseline_als(),
        Process.bg_removal_niter_fit(),
        # Process.bg_removal_niter_piecewisefit(),
    ]]

    for ele, preprocess in paras:
        num2label = {}
        for k in num2ele2label.keys():
            num2label[k] = num2ele2label[k][ele]
        name2label = {"neg": 0, "pos": 1}
        dataroot = os.path.join(dataroot_, ele)
        # modellist = [AlexNet_Sun, ResNet18, ResNet34]
        modellist = [AlexNet_Sun]
        db_cfg = dict(  # 数据集设置
            dataroot = dataroot,
            backEnd = ".csv",
            # backEnd = ".asc",
            t_v_t = [0.6, 0.2, 0.2],
            LoadCsvFile = readdatafunc,
            k_split = 4,
            transform = Process.process_series([  # 设置预处理流程
                Process.intorpolator(),
                Process.sg_filter(),
                preprocess,
                Process.norm_func(), ]
            ))
        recorddir = os.path.join(recordroot, ele)
        path2labelfunc = path2func_generator(num2label)
        train_modellist(dataroot, db_cfg = db_cfg, raman = raman, modellist = modellist, recorddir = recorddir,
                        path2labelfunc = path2labelfunc, sfname = "Raman_personwise_",n_iter = 5)


def main_onesrc_personwise():
    from samplewise2personwise import rename_files_between, rename_files_between_undo
    dataroot_ = os.path.join(projectroot, "data", "脑胶质瘤", "data_classified")
    rename_files_between(dataroot_, 3)
    try:
        main_one_datasrc(dataroot_)
    except:
        print("failed")
    finally:
        rename_files_between_undo(dataroot_, 3)


if __name__ == '__main__':
    main_onesrc_personwise()