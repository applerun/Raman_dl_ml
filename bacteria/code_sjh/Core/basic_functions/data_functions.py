import torch
import numpy

def pytorchlize(x):
    for i in range(3 - len(x.shape)):
        x = torch.unsqueeze(x, dim = len(x.shape) - 1)
    return x



def data2mean_std(spectrums: torch.Tensor or numpy.ndarray):
    """
    :param spectrums:a batch of spectrums[b,l] or [b,1,l]
    :returns spectrum_mean ,spectrum_up=spectrum_mean+std,spectrum_down=spectrum_mean-std
    """
    if type(spectrums) == numpy.ndarray:
        spectrums = torch.tensor(spectrums)
    if len(spectrums.shape) == 3 and spectrums.shape[1] == 1:
        spectrums = torch.squeeze(spectrums, dim = 1)
    elif len(spectrums.shape) > 2:
        raise AssertionError

    lenth = spectrums.shape[-1]
    batchsz = spectrums.shape[0]

    y_mean = torch.Tensor(lenth)
    y_up = torch.Tensor(lenth)
    y_down = torch.Tensor(lenth)
    for idx in range(lenth):
        p = spectrums[:, idx]
        mean = p.mean()
        std = p.std()
        y_mean[idx] = mean
        y_up[idx] = mean + std
        y_down[idx] = mean - std
    return y_mean, y_up, y_down