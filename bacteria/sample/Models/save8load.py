import os

import numpy
import torch
import onnx, onnxruntime

from bacteria.code_sjh.Core.basic_functions.path_func import getRootPath
from bacteria.code_sjh.models.CNN import AlexNet

project_root = getRootPath("Raman_dl_ml_code")
sample_root = os.path.join(project_root, "becteria", "sample")
save_root = r"res_temp"

if __name__ == '__main__':
    # torch.set_default_dtype(torch.double)
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    mod_path = os.path.join(save_root, "alexnet.pth")
    mod = AlexNet.AlexNet_Sun(torch.Tensor(3, 1, 512), 2)
    mod.eval()
    mod_s = torch.jit.script(mod)
    mod_IR = torch.jit.trace(mod, torch.Tensor(16, 1, 512))
    mod_s.save(mod_path)
    mod_onnx_path = os.path.join(save_root, "alexnet.onnx")

    # mod.save_model(mod_path)
    mod2 = torch.jit.load(mod_path)
    torch.onnx.export(mod, torch.Tensor(1, 1, 512), mod_onnx_path, opset_version = 11,
                      input_names = ['input'],
                      output_names = ['output'], dynamic_axes = {
            'input': {
                0: 'batch',
            },
            'output': {
                0: 'batch'
            }
        }
                      )
    mod_onnx = onnx.load(mod_onnx_path)
    onnx.checker.check_model(mod_onnx)
    session = onnxruntime.InferenceSession(mod_onnx_path, )
    s_input = dict(input = numpy.ones((16, 1, 512), dtype = "float32"))
    s_output = session.run(['output'], s_input)

    print(s_output[0].shape)
