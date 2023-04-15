import torch
def pytorchlize(x):
    for i in range(3 - len(x.shape)):
        x = torch.unsqueeze(x, dim = len(x.shape) - 1)
    return x