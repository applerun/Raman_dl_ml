import os, shutil

current_dir = os.path.dirname(__file__)
coderoot = os.path.dirname(os.path.split(os.path.split(__file__)[0])[0])
coderoot = os.path.dirname(coderoot)
projectroot = os.path.split(coderoot)[0]
dst_root = os.path.join(projectroot, "data", "脑胶质瘤", "data_used")
dataroot_batch1 = os.path.join(projectroot, "data", "脑胶质瘤", "batch1")
dataroot_batch2 = os.path.join(projectroot, "data", "脑胶质瘤", "batch2")
dataroot_batch3 = os.path.join(projectroot, "data", "脑胶质瘤", "batch3")


def datafileSortbyPoint(
        dirpath = dataroot_batch1,
        files_per_dir = None,
        dir_inf = "point",
        backend = ".csv",
        maxPoints = 10,

):
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
        dir = os.path.join(dirpath, dir_inf + str(dir_count))
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

    for i in range(3):
        batch = os.path.join(dst_root, "batch{}".format(i + 1))
        for dir in os.listdir(batch):
            datafileSortbyPoint(os.path.join(batch, dir), None if i == 0 else 3)

# datafileSortbyPoint(os.path.join(data_root, "Abnorm", "牛淑雨"), 3)
