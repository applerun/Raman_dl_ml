import csv

import numpy

from bacteria.code_sjh.utils.RamanData import RamanDatasetCore, Raman_dirwise
from bacteria.code_sjh.models import BasicModule
from bacteria.code_sjh.utils.Validation.validation import grad_cam
import os
import torch
import shutil


def copy_filewise_classify(db: RamanDatasetCore,
                           model: BasicModule,
                           dst: str,
                           device: torch.device = None):
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = db.root
    if not os.path.isdir(dst):
        os.makedirs(dst)
    model = model.to(device)
    label2name = db.label2name()
    for i in range(len(db)):
        raman, label = db[i]
        raman = torch.unsqueeze(raman, dim = 0).to(device)

        file = db.RamanFiles[i]
        src = os.path.join(root, file)
        filename = file.split(os.sep)[-1]
        with torch.no_grad():
            logits = model(raman)  # [1,n_c]

        pred = torch.squeeze(logits.argmax(dim = 1)).item()
        true_name = label2name[label.item()]
        pred_name = label2name[pred]
        dst_dir = os.path.join(dst, true_name + "2" + pred_name)
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        dst_p = os.path.join(dst_dir, filename)
        shutil.copy(src, dst_p)


def cam_output_filewise(db: RamanDatasetCore,
                        model: BasicModule,
                        dst: str,
                        device: torch.device = None):
    model.train()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # root = db.root
    if not os.path.isdir(dst):
        os.makedirs(dst)
    model = model.to(device)
    # label2name = db.label2name()
    if not os.path.isdir(dst):
        os.makedirs(dst)
    for i in range(len(db)):
        raman, label = db[i]
        raman = torch.unsqueeze(raman, dim = 0).to(device)
        file = db.RamanFiles[i]
        filename = file.split(os.sep)[-1][:-4] + ".cam.csv"
        cam = grad_cam(model, raman, label = None)
        dst_p = os.path.join(dst, filename)
        numpy.savetxt(dst_p, cam, delimiter = ",")


def report_dirwise_classify(db: Raman_dirwise,  # raman_dir -> root/class/dir/filename
                            model: BasicModule,
                            dst: str,
                            device: torch.device = None,
                            label2name = None,
                            ):
    """
    生成每个文件夹对应的分类结果报告：
        流程：
            加载网络
            对db的每个数据进行分类（root/class/sample/filename）
            统计每个sample文件夹下各个文件（root/class/sample）的分类结果

    @param db:
    @param model:
    @param dst:
    @param device:
    @return:
    """

    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = db.root
    if not os.path.isdir(dst):
        os.makedirs(dst)
    model = model.to(device)
    if label2name is None:
        label2name = db.label2name()
    sample2label = db.sample2label()
    sample2data = db.get_data_sorted_by_sample()
    f = open(dst,"w",newline = "")
    writer = csv.writer(f)
    header = ["sample"]
    for label in label2name.keys():
        header.append(label2name[label])
    header.append("sum")
    writer.writerow(header)
    for sample in sample2data.keys():
        res = [sample] + [0] * (len(header))
        raman = sample2data[sample]
        with torch.no_grad():
            logits = model(raman)  # [num_sample,n_c]
        pred = torch.squeeze(logits.argmax(dim = 1)).item()  # [num_sample]
        for i in range(len(pred)):
            res[pred + 1] += 1
            res[-1] += 1
        assert sum(res[1:-1]) == res[-1]
        writer.writerow(res)
