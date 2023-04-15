import numpy
import copy


def getRamanFromFile(wavelengthstart = 400,
                     wavelengthend = 1800,
                     dataname2idx = None,
                     delimeter = None):
    if wavelengthend < wavelengthstart:
        wavelengthstart, wavelengthend = wavelengthend, wavelengthstart


    def func(filepath: str,
             delimeter = delimeter, dataname2idx = dataname2idx):
        dataname2idx = copy.deepcopy(dataname2idx)
        if dataname2idx is None:
            dataname2idx = {}
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
                if not len(line):
                    continue
                data = line.split(delimeter)

                if data[0] in ["ROI", "Wavelength", "Column", "Intensity"]:
                    if header is None:
                        header = data
                        dataname2idx["Wavelength"] = header.index("Wavelength")
                        dataname2idx["Intensity"] = header.index("Intensity")
                    continue
                try:
                    wavelength = float(data[dataname2idx["Wavelength"]])
                    intense = float(data[dataname2idx["Intensity"]])
                except:
                    # print(filepath, ":", data, ",delimeter:", delimeter,"unused line :",line)
                    continue
                if wavelengthstart < wavelength and wavelength < wavelengthend:
                    Ramans.append(intense)
                    Wavelengths.append(wavelength)
                elif wavelength > wavelengthend:
                    break
        Ramans = numpy.array([Ramans])
        Wavelengths = numpy.array(Wavelengths)
        return Ramans, Wavelengths

    return func


# 请将combined文件后缀前加上减号" - "
def LoadCombinedFile(filename: str,
                     encoding = "utf-8-sig"):
    """
    本函数用于读取数据
    :param encoding:
    :param filename: data file name
    :return:
    """
    if not filename.endswith(".csv"):
        filename += ".csv"

    with open(filename, "r", encoding = encoding) as f:
        lines = f.readlines()

        for line in lines:
            data = [float(i) for i in line.split(",")]  # 将字符串转化为浮点数
            yield data
    return None
