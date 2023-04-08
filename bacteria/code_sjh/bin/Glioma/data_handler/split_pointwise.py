import os, shutil

current_dir = os.path.dirname(__file__)
coderoot = os.path.dirname(os.path.split(os.path.split(__file__)[0])[0])
coderoot = os.path.dirname(coderoot)
projectroot = os.path.split(coderoot)[0]


def datafileSortbyPoint(
        dirpath,
        files_per_dir = None,
        dir_inf = "point",
        backend = ".csv",
        maxPoints = 11,

):
    """
    将文件夹下的数据按一定的策略分成各个子文件夹内（同一个点的光谱放在一个子文件夹里）
    @param dirpath: 文件夹路径
    @param files_per_dir: 每个子文件夹包含的文件数量
    @param dir_inf: 每个子文件夹的命名
    @param backend: 需要进行整理的文件后缀
    @param maxPoints: 子文件夹最多生成的数量
    @return:
    """
    file_count = 0

    mode = 0 if files_per_dir is None else 1  # 0:根据文件名分割，1：根据每个点的个数分割
    for files in os.listdir(dirpath):
        abs_file = os.path.join(dirpath, files)
        if not os.path.isfile(abs_file) or not files.endswith(backend):
            continue
        if "995" in files:
            continue
        if mode == 1:
            dir_count = file_count // files_per_dir
        else:
            dir_count = int(files.split(" ")[0])
        if dir_count == maxPoints:
            return
        while os.path.isdir(os.path.join(dirpath, dir_inf + "_" + str(dir_count))):
            dir_count += 1
        dir = os.path.join(dirpath, dir_inf + "_" + str(dir_count))
        if not os.path.isdir(dir):
            os.makedirs(dir)
        file_count += 1

        dst_file = os.path.join(dir, files)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        os.rename(abs_file, dst_file)
    # shutil.copy(abs_file, dst_file)
    return


if __name__ == '__main__':
    # dst_root = os.path.join(projectroot, "data", "脑胶质瘤", "data_used")
    # dataroot_batch1 = os.path.join(projectroot, "data", "脑胶质瘤", "batch1")
    # dataroot_batch2 = os.path.join(projectroot, "data", "脑胶质瘤", "batch2")
    # dataroot_batch3 = os.path.join(projectroot, "data", "脑胶质瘤", "batch3")
    # for i in range(3):
    #     batch = os.path.join(dst_root, "batch{}".format(i + 1))
    #     for dir in os.listdir(batch):
    #         datafileSortbyPoint(os.path.join(batch, dir), None if i == 0 else 3)
    data_root = os.path.join(projectroot, "data", "脑胶质瘤", "data_indep_unlabeled", "unlabeled")
    for tissue_dir in os.listdir(data_root):
        datafileSortbyPoint(os.path.join(data_root, tissue_dir), 3, maxPoints = 99)
