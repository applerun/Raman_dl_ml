try:
    from bacteria.code_sjh.models.CNN.AlexNet import AlexNet_Sun
    from bacteria.code_sjh.utils.RamanData import Raman, getRamanFromFile, Raman_dirwise
    from bacteria.code_sjh.utils.Validation.validation import *
    from bacteria.code_sjh.utils.Validation.visdom_utils import *
    from bacteria.code_sjh.utils.iterator import train
except:
    sys.path.append(coderoot)
    from utils.RamanData import Raman, getRamanFromFile, Raman_dirwise
    from models.CNN.AlexNet import AlexNet_Sun
    from utils.Validation.validation import *
    from utils.Validation.visdom_utils import *
    from utils.iterator import train
from bacteria.code_sjh.config.data_Config.csvdata import *

# readdatafunc = getRamanFromFile(wavelengthstart = 400,wavelengthend = 1800,delimeter = delimeter,dataname2idx = dataformat)


from scipy import interpolate

backend = data_config["backend"]


readdatafunc0 = getRamanFromFile(wavelengthstart = 596, wavelengthend = 1802, delimeter = data_config["delimeter"],
                                 dataname2idx = data_config["dataname2idx"])
def readdatafunc1(filepath):
    R, X = readdatafunc0(filepath)
    f = interpolate.interp1d(X, R, kind = "cubic")
    newX = numpy.linspace(600, 1800, 760)
    newR = f(newX)
    return newR, newX

train_cfg = dict(
    lr = 0.0001,
    epochs = 60,
)

database_cfg = dict(
    dataroot = dataroot,
    backEnd = backend,
    t_v_t = [0.6, 0.2, 0.2],
    LoadCsvFile = readdatafunc1,
    k_split = 4
)

