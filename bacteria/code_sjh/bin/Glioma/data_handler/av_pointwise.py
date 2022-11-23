import os
import warnings
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def getRamanFromFile(wavelengthstart = 0,
                     wavelengthend = 100000,
                     dataname2idx = None,
                     delimeter = None):
    if wavelengthend < wavelengthstart:
        wavelengthstart, wavelengthend = wavelengthend, wavelengthstart

    def func(filepath: str,
             delimeter = delimeter,
             in_dataname2idx = dataname2idx):
        if in_dataname2idx is None:
            in_dataname2idx = {}
        else:
            in_dataname2idx = copy.deepcopy(in_dataname2idx)
        Ramans = []
        Wavelengths = []
        if delimeter is None:
            if filepath.endswith(".csv"):
                delimeter = ","
            elif filepath.endswith(".asc"):
                delimeter = "\t"

        with open(filepath, "r") as f:
            lines = f.readlines()
            header = None
            for line in lines:
                line = line.strip()
                data = line.split(delimeter)

                if data[0] in ["ROI", "Wavelength", "Column", "Intensity"]:
                    if header is None:
                        header = data
                        in_dataname2idx["Wavelength"] = header.index("Wavelength")
                        in_dataname2idx["Intensity"] = header.index("Intensity")
                    continue
                try:
                    wavelength = float(data[in_dataname2idx["Wavelength"]])
                    intense = float(data[in_dataname2idx["Intensity"]])
                except:
                    print(filepath, ":", data, ",delimeter:", delimeter)
                    raise ValueError
                if wavelengthstart < wavelength and wavelength < wavelengthend:
                    Ramans.append(intense)
                    Wavelengths.append(wavelength)
                elif wavelength > wavelengthend:
                    break
        assert all([1 == x for x in Ramans]) is False
        Ramans = np.array([Ramans])
        Wavelengths = np.array(Wavelengths)
        return Ramans, Wavelengths
    return func


def linearizeByterminal(slit: np.ndarray):
    shape = copy.deepcopy(slit.shape)
    l = len(slit.squeeze())
    for i in range(l):
        slit[i] = (l - i - 1) / (l - 1) * slit[0] + i / (l - 1) * slit[-1]
    return slit.reshape(shape)


def smooth(data: np.ndarray,
           max_ = 1000,
           span = 3):
    shape = copy.deepcopy(data.shape)
    if len(data.shape) == 2:
        data = data.squeeze()

    for i in range(len(data)):
        slit = data[max(0, i - span):min(len(data) - 1, i + span)]
        if data[i] - slit.min() > max_:
            data[max(0, i - span):min(len(data) - 1, i + span)] = linearizeByterminal(
                data[max(0, i - span):min(len(data) - 1, i + span)])
    return data.reshape(shape)


# if __name__ == '__main__':
# 	data = np.array([[1., 2., 12000., 3., 4., 5.]])
# 	data = smooth(data)
# 	print(data)

def refile(src_file, readRaman, dst_file = None, xs = None):
    raman, wavelength = readRaman(src_file)
    if raman[0][0] == 1:
        return
    if xs is not None:
        wavelength = xs
    raman = smooth(raman)
    if dst_file is None:
        dst_file = src_file
    if os.path.isfile(dst_file):
        os.remove(dst_file)
    np.savetxt(dst_file, np.vstack((wavelength, raman)).T, header = "Wavelength,Intensity", delimiter = ",",
               comments = "")


def dir2file_av(src_dir,
                readRaman,
                dst_file = None,
                backend = ".csv",
                ):
    if not os.path.isdir(src_dir):
        raise AssertionError("Not A Dir:" + src_dir)
    if dst_file is None:
        dst_file = src_dir + ".csv"
    wavelengths_all = None
    ramans_all = None
    for files in os.listdir(src_dir):
        abs_file = os.path.join(src_dir, files)
        if not files.endswith(backend) or not os.path.isfile(abs_file):
            continue
        raman, wavelength = readRaman(abs_file)
        raman = smooth(raman)
        ramans_all = raman if ramans_all is None else np.vstack((ramans_all, raman))
        if wavelengths_all is None:
            wavelengths_all = wavelength
        elif np.abs(wavelengths_all - wavelength).sum() > 0.01:
            warnings.warn("too different between the first file and {}".format(abs_file))
    ramans_all = np.average(ramans_all, axis = 0)
    # print(ramans_all)
    if os.path.isfile(dst_file):
        os.remove(dst_file)
    np.savetxt(dst_file, np.vstack((wavelengths_all, ramans_all)).T, header = "Wavelength,Intensity", delimiter = ",",
               comments = "")


def file2plot(src,
              dst,
              readRaman, ):
    raman, wavelength = readRaman(src)
    plt.plot(np.squeeze(wavelength), np.squeeze(raman), )
    plt.savefig(dst)
    plt.close()


if __name__ == '__main__':
    classes = ["batch1", "batch2", "batch3"]
    # dir_prefix = "point"
    current_dir = os.path.dirname(__file__)
    data_root = "../../../../data/脑胶质瘤/data_reformed/"
    xs = None
    for c in classes:
        if c == "batch2":
            readRaman = getRamanFromFile(0, 4000, dataname2idx = {"Wavelength": 2, "Intensity": 5})
        elif c == "batch1":
            readRaman = getRamanFromFile(0, 4000, dataname2idx = {"Wavelength": 5, "Intensity": 4})
        else:
            readRaman = getRamanFromFile(0, 4000, dataname2idx = {"Wavelength": 2, "Intensity": 5})
        class_root = os.path.join(data_root, c)
        for dirs in os.listdir(class_root):
            abs_dir = os.path.join(class_root, dirs)
            if not os.path.isdir(abs_dir):
                continue
            for pointdir in os.listdir(abs_dir):
                abs_pointdir = os.path.join(abs_dir, pointdir)
                if not os.path.isdir(abs_pointdir):
                    # or not pointdir.startswith(dir_prefix):
                    continue
                dst_file = abs_pointdir + ".csv"
                for raman_file in os.listdir(abs_pointdir):
                    abs_raman_file = os.path.join(abs_pointdir, raman_file)
                    refile(abs_raman_file, readRaman)

                dir2file_av(abs_pointdir, readRaman, dst_file)
                # file2plot(dst_file, dst_file[:-4] + ".png", readRaman)

            # for frames in os.listdir(abs_pointdir):
            # 	abs_frame = os.path.join(abs_pointdir, frames)
            # 	file2plot(abs_frame, abs_frame[:-4] + ".png")
