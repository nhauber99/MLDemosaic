import copy
import os

import torch

from Config import checkpoints_dir, DEVICE
from Model import DemosaicModel

if __name__ == "__main__":
    model = DemosaicModel()
    model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
    model = model.to(DEVICE).eval()

    example = (torch.rand(1, 1, 512, 512).cuda(),)

    traced_script_module = torch.jit.script(model)
    torch.onnx.export(traced_script_module, example, 'model.onnx', input_names=["in"],
                      output_names=["out"], verbose=False, export_params=True, opset_version=17)
