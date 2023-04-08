import os
import shutil

datafile = r"...\01 ZLX 739785 T1\point?(.csv)"
dstfile = r"...\01\ZLX 739785 T1 point?(.csv)"


def rename(src, spliter = " "):
    dir_src = os.path.dirname(src)  # ...\01 ZLX 739785 T1
    dir_src_base = os.path.basename(dir_src)  # 01 ZLX 739785 T1
    dir_dst_base, base_dst = dir_src_base.split(spliter, 1)  # 01 , ZLX 739785 T1
    base_dst = spliter.join([base_dst, os.path.basename(src)])  # ZLX 739785 T1 point?(.csv)
    dst = os.path.join(os.path.dirname(dir_src), dir_dst_base, base_dst)  # ...\01 \ZLX 739785 T1 point?(.csv)
    return dst


def rename_undo(src, spliter = " "):
    infolist = os.path.basename(src).split(spliter)
    dir_dst = spliter.join([os.path.dirname(src)] + infolist[:-1])
    base_dst = infolist[-1]
    return os.path.join(dir_dst, base_dst)


def rename_files_between(root, depth, warning = False):
    for r, d, f in os.walk(root):
        f = d + f
        d = len(r.replace(root, "").split(os.sep)) - 1
        if d != depth:
            if warning:
                print("depth {} != min_depth {}, ignore dir {}".format(
                    d, depth, r))
            continue
        for file in f:
            file = os.path.join(r,file)
            dst = rename(file)

            if not os.path.isdir(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))
            os.rename(file,dst)
        os.rmdir(r)
    # for file in files:
    #     dst = rename(file)
    #     os.rename(file,dst)

def rename_files_between_undo(root, depth, warning = False):
    for r, d, f in os.walk(root):
        f = d + f
        d = len(r.replace(root, "").split(os.sep)) - 1
        if d != depth:
            if warning:
                print("depth {} != min_depth {}, ignore dir {}".format(
                    d, depth, r))
            continue
        for file in f:
            file = os.path.join(r, file)
            dst = rename_undo(file)

            if not os.path.isdir(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))
            os.rename(file, dst)
        os.rmdir(r)
if __name__ == '__main__':
    print(datafile)
    dstfile = rename(datafile)
    print(dstfile)
    datafile2 = rename_undo(dstfile)
    print(datafile2)
    dataroot_src = r"D:\myPrograms\pythonProject\Raman_dl_ml\bacteria\data\脑胶质瘤\data_rename_test - 副本"
    dataroot = r"D:\myPrograms\pythonProject\Raman_dl_ml\bacteria\data\脑胶质瘤\data_rename_test"
    if os.path.isdir(dataroot):
        shutil.rmtree(dataroot)
    shutil.copytree(dataroot_src,dataroot)
    rename_files_between(dataroot,3)
    rename_files_between_undo(dataroot,3)
