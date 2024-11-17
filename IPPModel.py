import ctypes
from enum import Enum

import torch
from torch import nn
import PyIPP


class DemosaicMethod(Enum):
    AHD = 0  # https://www.intel.com/content/www/us/en/docs/ipp/developer-reference/2021-7/demosaicahd.html
    VNG = 1  # https://www.intel.com/content/www/us/en/docs/ipp/developer-reference/2021-7/cfatobgra.html
    Legacy = 2  # https://www.intel.com/content/www/us/en/docs/ipp/developer-reference/2021-7/cfatorgb.html


class IPPModel(nn.Module):
    def __init__(self, method: DemosaicMethod = DemosaicMethod.AHD):
        super(IPPModel, self).__init__()
        self.method = method

    def forward(self, x):
        device = x.device
        x = x.cpu()
        xl = torch.unbind(x, dim=0)
        results = torch.zeros((x.shape[0], 3, x.shape[2], x.shape[3]))
        for i, image in enumerate(xl):
            result = torch.zeros((3, image.shape[1], image.shape[2])).contiguous()
            temp = (image * (2 ** 16 - 0.5)).clip(0, 2 ** 16 - 0.001).contiguous()
            PyIPP.Demosaic(
                ctypes.c_void_p(temp.data_ptr()).value,
                ctypes.c_void_p(result.data_ptr()).value,
                3,
                image.shape[1],
                image.shape[2],
                int(self.method.value),
                1
            )
            results[i] = result / (2 ** 16 - 0.5)
        return results.to(device).clip(0, 1)


if __name__ == '__main__':
    model = IPPModel()
    y = model(torch.randn(1, 1, 256, 256))
    print(y.shape)
