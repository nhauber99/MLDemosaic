import torch
from torch import nn

from Config import DEVICE


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels, out_channels, growth_channels) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, out_channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

        self.init_weights(self.conv1)
        self.init_weights(self.conv2)
        self.init_weights(self.conv3)
        self.init_weights(self.conv4)
        self.init_weights(self.conv5)

    def init_weights(self, conv):
        nn.init.kaiming_normal_(conv.weight)
        conv.weight.data *= 0.2
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def forward(self, x):
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))

        x = torch.mul(out5, 0.2)
        x = torch.add(x, identity)

        return x


class ResidualResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, out_channels: int, growth_channels: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, out_channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, out_channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, out_channels, growth_channels)

    def forward(self, x):
        identity = x

        x = self.rdb1(x)
        x = self.rdb2(x)
        x = self.rdb3(x)

        x = torch.mul(x, 0.2)
        x = torch.add(x, identity)

        return x


class BilinearModel(nn.Module):
    def __init__(self):
        super(BilinearModel, self).__init__()
        self.bilinear_conv = nn.Conv2d(in_channels=4, out_channels=12, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        w = torch.zeros((3, 2, 2, 2, 2, 3, 3))
        # c oy ox iy ix
        w[0, 0, 0, 0, 0] = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        w[1, 0, 0, 0, 1] = torch.tensor([[0, 0, 0], [0.25, 0.25, 0], [0, 0, 0]])
        w[1, 0, 0, 1, 0] = torch.tensor([[0, 0.25, 0], [0, 0.25, 0], [0, 0, 0]])
        w[2, 0, 0, 1, 1] = torch.tensor([[0.25, 0.25, 0], [0.25, 0.25, 0], [0, 0, 0]])

        w[0, 0, 1, 0, 0] = torch.tensor([[0, 0, 0], [0, 0.5, 0.5], [0, 0, 0]])
        w[1, 0, 1, 0, 1] = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        w[2, 0, 1, 1, 1] = torch.tensor([[0, 0.5, 0], [0, 0.5, 0], [0, 0, 0]])

        w[0, 1, 0, 0, 0] = torch.tensor([[0, 0, 0], [0, 0.5, 0], [0, 0.5, 0]])
        w[1, 1, 0, 1, 0] = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        w[2, 1, 0, 1, 1] = torch.tensor([[0, 0, 0], [0.5, 0.5, 0], [0, 0, 0]])

        w[0, 1, 1, 0, 0] = torch.tensor([[0, 0, 0], [0, 0.25, 0.25], [0, 0.25, 0.25]])
        w[1, 1, 1, 0, 1] = torch.tensor([[0, 0, 0], [0, 0.25, 0], [0, 0.25, 0]])
        w[1, 1, 1, 1, 0] = torch.tensor([[0, 0, 0], [0, 0.25, 0.25], [0, 0, 0]])
        w[2, 1, 1, 1, 1] = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        self.bilinear_conv.weight.data = w.reshape(12, 4, 3, 3)
        self.bilinear_conv.weight.requires_grad = False

    def forward(self, x):
        y = torch.nn.functional.pixel_unshuffle(x, 2)
        y = self.bilinear_conv(y)
        return torch.nn.functional.pixel_shuffle(y, 2)


class DemosaicModel(nn.Module):
    def __init__(self, gamma2=False):
        super(DemosaicModel, self).__init__()
        self.gamma2 = gamma2
        self.bilinear = BilinearModel()

        self.conv0 = nn.Sequential(
            nn.PixelUnshuffle(2),  # transforms the 4 pixels per bayer patch to 4 channels at a single position instead
            nn.Conv2d(4, 32, 7, 1, 3)
        )
        self.rrdb1 = ResidualResidualDenseBlock(32, 32, 16)
        self.rrdb2 = ResidualResidualDenseBlock(32, 32, 16)
        self.rrdb3 = ResidualResidualDenseBlock(32, 32, 16)

        self.deconv0 = nn.Sequential(
            nn.Conv2d(32, 4 * 32, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 16, 3, 1, 1),
            torch.nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 3, 3, 1, 1)
        )

    def forward(self, x):
        bilinear = self.bilinear(x)
        # model is run on the image with gamma=2 as it's more perceptually uniform than the linear image
        tmp = self.conv0(torch.sqrt(x))
        tmp = self.rrdb1(tmp)
        tmp = self.rrdb2(tmp)
        tmp = self.rrdb3(tmp)
        tmp = self.deconv0(tmp)
        res = torch.sqrt_(bilinear) + tmp
        return res if self.gamma2 else torch.square_(res)


if __name__ == '__main__':
    model = DemosaicModel().to(DEVICE)
    y = model(torch.randn(1, 1, 256, 256).to(DEVICE))
    print(y.shape)
