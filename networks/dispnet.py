import torch
import torch.nn as nn
from math import ceil
from .dispnet_submodules import *


flag_bias_t = True
flag_bn = False
activefun_t = nn.ReLU(inplace=True)


class Dispnet_v1(nn.Module):
    def __init__(self, maxdisparity=192):
        super(Dispnet_v1, self).__init__()
        self.name = "Dispnet_v1"
        self.D = maxdisparity
        self.delt = 1e-6
        self.count_levels = 7

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = conv2d_bn(6, 64, kernel_size=7, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv2 = conv2d_bn(64, 128, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3a = conv2d_bn(128, 256, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3b = conv2d_bn(256, 256, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv4a = conv2d_bn(256, 512, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv4b = conv2d_bn(512, 512, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)

        self.pr4 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

        self.deconv3 = deconv2d_bn(512, 128, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv3 = conv2d_bn(385, 128, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.deconv2 = deconv2d_bn(128, 64, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv2 = conv2d_bn(193, 64, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.deconv1 = deconv2d_bn(64, 32, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv1 = conv2d_bn(97, 32, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        net_init(self)
        for m in [self.pr4, self.pr3, self.pr2, self.pr1]:
            m.weight.data = m.weight.data*0.1

    def forward(self, imL, imR, mode="train"):
        assert imL.shape == imR.shape
        maxD = max(self.D, imL.shape[-1])
        out = []
        out_scale = []

        x = torch.cat([imL, imR], dim=1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3a = self.conv3a(conv2)
        conv3b = self.conv3b(conv3a)
        conv4a = self.conv4a(conv3b)
        conv4b = self.conv4b(conv4a)

        pr4 = self.pr4(conv4b)
        #out.insert(0, pr4)
        #out_scale.insert(0, 4)
        pr4_up = self.upsample(pr4)

        deconv3 = self.deconv3(conv4b)
        iconv3 = self.iconv3(torch.cat([deconv3, pr4_up, conv3b], dim = 1))
        pr3 = self.pr3(iconv3)
        #out.insert(0, pr3)
        #out_scale.insert(0, 3)
        pr3_up = self.upsample(pr3)

        deconv2 = self.deconv2(iconv3)
        iconv2 = self.iconv2(torch.cat([deconv2, pr3_up, conv2], dim = 1))
        pr2 = self.pr2(iconv2)
        #out.insert(0, pr2)
        #out_scale.insert(0, 2)
        pr2_up = self.upsample(pr2)

        deconv1 = self.deconv1(iconv2)
        iconv1 = self.iconv1(torch.cat([deconv1, pr2_up, conv1], dim = 1))
        pr1 = self.pr1(iconv1)
        #out.insert(0, pr1)
        #out_scale.insert(0, 1)
        pr1_up = self.upsample(pr1)

        #pr0 = pr1[:, :, :imL.shape[-2], :imL.shape[-1]]
        #out.insert(0, pr0)
        #out_scale.insert(0, 0)
        if(mode == "test"): out[-1] = out[-1].clamp(self.delt, maxD)

        #return out_scale, out
        return [pr1_up, pr1, pr2, pr3, pr4]


class Dispnet_v2(nn.Module):
    def __init__(self, maxdisparity=192):
        super(Dispnet_v2, self).__init__()
        self.name = "Dispnet_v2"
        self.D = maxdisparity
        self.delt = 1e-6
        self.count_levels = 7

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = conv2d_bn(2, 64, kernel_size=7, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv2 = conv2d_bn(64, 128, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3a = conv2d_bn(128, 256, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3b = conv2d_bn(256, 256, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv4a = conv2d_bn(256, 512, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv4b = conv2d_bn(512, 512, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)

        self.pr4 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

        self.deconv3 = deconv2d_bn(512, 128, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv3 = conv2d_bn(385, 128, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.deconv2 = deconv2d_bn(128, 64, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv2 = conv2d_bn(193, 64, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.deconv1 = deconv2d_bn(64, 32, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv1 = conv2d_bn(97, 32, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        net_init(self)
        for m in [self.pr4, self.pr3, self.pr2, self.pr1]:
            m.weight.data = m.weight.data*0.1

    def forward(self, imL, imR, mode="train"):
        assert imL.shape == imR.shape
        maxD = max(self.D, imL.shape[-1])
        out = []

        pad_numbers = (0, ceil(imL.shape[3] / 16.) * 16 - imL.shape[3], 0, ceil(imL.shape[2] / 16.) * 16 - imL.shape[2])

        imL = nn.functional.pad(imL, pad_numbers)
        imR = nn.functional.pad(imR, pad_numbers)

        x = torch.cat([imL, imR], dim=1)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3a = self.conv3a(conv2)
        conv3b = self.conv3b(conv3a)
        conv4a = self.conv4a(conv3b)
        conv4b = self.conv4b(conv4a)

        pr4 = self.pr4(conv4b)
        #out.insert(0, pr4)
        #out_scale.insert(0, 4)
        pr4_up = self.upsample(pr4)

        deconv3 = self.deconv3(conv4b)
        iconv3 = self.iconv3(torch.cat([deconv3, pr4_up, conv3b], dim = 1))
        pr3 = self.pr3(iconv3)
        #out.insert(0, pr3)
        #out_scale.insert(0, 3)
        pr3_up = self.upsample(pr3)

        deconv2 = self.deconv2(iconv3)
        iconv2 = self.iconv2(torch.cat([deconv2, pr3_up, conv2], dim = 1))
        pr2 = self.pr2(iconv2)
        #out.insert(0, pr2)
        #out_scale.insert(0, 2)
        pr2_up = self.upsample(pr2)

        deconv1 = self.deconv1(iconv2)
        iconv1 = self.iconv1(torch.cat([deconv1, pr2_up, conv1], dim = 1))
        pr1 = self.pr1(iconv1)
        #out.insert(0, pr1)
        #out_scale.insert(0, 1)
        pr1_up = self.upsample(pr1)

        #pr0 = pr1[:, :, :imL.shape[-2], :imL.shape[-1]]
        # out.insert(0, pr0)
        # out_scale.insert(0, 0)
        if(mode == "test"): out[-1] = out[-1].clamp(self.delt, maxD)

        # return out_scale, out
        # return [pr1_up, pr1, pr2, pr3, pr4]
        return pr1_up[:, :, :imL.shape[2]-pad_numbers[3], :imR.shape[3]-pad_numbers[1]]


class Dispnet_4to2(nn.Module):
    def __init__(self, maxdisparity=192):
        super(Dispnet_4to2, self).__init__()
        self.name = "Dispnet_v2"
        self.D = maxdisparity
        self.delt = 1e-6
        self.count_levels = 7

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = conv2d_bn(4, 64, kernel_size=7, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv2 = conv2d_bn(64, 128, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3a = conv2d_bn(128, 256, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3b = conv2d_bn(256, 256, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv4a = conv2d_bn(256, 512, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv4b = conv2d_bn(512, 512, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)

        self.pr4 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

        self.deconv3 = deconv2d_bn(512, 128, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv3 = conv2d_bn(385, 128, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.deconv2 = deconv2d_bn(128, 64, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv2 = conv2d_bn(193, 64, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.deconv1 = deconv2d_bn(64, 32, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv1 = conv2d_bn(97, 32, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr1 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)

        net_init(self)
        for m in [self.pr4, self.pr3, self.pr2, self.pr1]:
            m.weight.data = m.weight.data*0.1

    def forward(self, imL, gradL, imR, gradR, mode="train"):
        assert imL.shape == imR.shape
        assert imL.shape == gradL.shape
        assert imR.shape == gradR.shape
        maxD = max(self.D, imL.shape[-1])
        out = []

        pad_numbers = (0, ceil(imL.shape[3] / 16.) * 16 - imL.shape[3], 0, ceil(imL.shape[2] / 16.) * 16 - imL.shape[2])

        imL = nn.functional.pad(imL, pad_numbers)
        gradL = nn.functional.pad(gradL, pad_numbers)
        imR = nn.functional.pad(imR, pad_numbers)
        gradR = nn.functional.pad(gradR, pad_numbers)

        x = torch.cat([imL, gradL, imR, gradR], dim=1)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3a = self.conv3a(conv2)
        conv3b = self.conv3b(conv3a)
        conv4a = self.conv4a(conv3b)
        conv4b = self.conv4b(conv4a)

        pr4 = self.pr4(conv4b)
        #out.insert(0, pr4)
        #out_scale.insert(0, 4)
        pr4_up = self.upsample(pr4)

        deconv3 = self.deconv3(conv4b)
        iconv3 = self.iconv3(torch.cat([deconv3, pr4_up, conv3b], dim = 1))
        pr3 = self.pr3(iconv3)
        #out.insert(0, pr3)
        #out_scale.insert(0, 3)
        pr3_up = self.upsample(pr3)

        deconv2 = self.deconv2(iconv3)
        iconv2 = self.iconv2(torch.cat([deconv2, pr3_up, conv2], dim = 1))
        pr2 = self.pr2(iconv2)
        #out.insert(0, pr2)
        #out_scale.insert(0, 2)
        pr2_up = self.upsample(pr2)

        deconv1 = self.deconv1(iconv2)
        iconv1 = self.iconv1(torch.cat([deconv1, pr2_up, conv1], dim = 1))
        pr1 = self.pr1(iconv1)
        #out.insert(0, pr1)
        #out_scale.insert(0, 1)
        pr1_up = self.upsample(pr1)

        #pr0 = pr1[:, :, :imL.shape[-2], :imL.shape[-1]]
        # out.insert(0, pr0)
        # out_scale.insert(0, 0)
        if(mode == "test"): out[-1] = out[-1].clamp(self.delt, maxD)

        # return out_scale, out
        # return [pr1_up, pr1, pr2, pr3, pr4]
        return pr1_up[:, :, :imL.shape[2]-pad_numbers[3], :imR.shape[3]-pad_numbers[1]]


class Dispnet_4to1(nn.Module):
    def __init__(self, maxdisparity=192):
        super(Dispnet_4to1, self).__init__()
        self.name = "Dispnet_v2"
        self.D = maxdisparity
        self.delt = 1e-6
        self.count_levels = 7

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = conv2d_bn(4, 64, kernel_size=7, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv2 = conv2d_bn(64, 128, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3a = conv2d_bn(128, 256, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3b = conv2d_bn(256, 256, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv4a = conv2d_bn(256, 512, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv4b = conv2d_bn(512, 512, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)

        self.pr4 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

        self.deconv3 = deconv2d_bn(512, 128, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv3 = conv2d_bn(385, 128, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.deconv2 = deconv2d_bn(128, 64, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv2 = conv2d_bn(193, 64, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.deconv1 = deconv2d_bn(64, 32, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv1 = conv2d_bn(97, 32, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        net_init(self)
        for m in [self.pr4, self.pr3, self.pr2, self.pr1]:
            m.weight.data = m.weight.data*0.1

    def forward(self, imL, gradL, imR, gradR, mode="train"):
        assert imL.shape == imR.shape
        assert imL.shape == gradL.shape
        assert imR.shape == gradR.shape
        maxD = max(self.D, imL.shape[-1])
        out = []

        pad_numbers = (0, ceil(imL.shape[3] / 16.) * 16 - imL.shape[3], 0, ceil(imL.shape[2] / 16.) * 16 - imL.shape[2])

        imL = nn.functional.pad(imL, pad_numbers)
        gradL = nn.functional.pad(gradL, pad_numbers)
        imR = nn.functional.pad(imR, pad_numbers)
        gradR = nn.functional.pad(gradR, pad_numbers)

        x = torch.cat([imL, gradL, imR, gradR], dim=1)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3a = self.conv3a(conv2)
        conv3b = self.conv3b(conv3a)
        conv4a = self.conv4a(conv3b)
        conv4b = self.conv4b(conv4a)

        pr4 = self.pr4(conv4b)
        #out.insert(0, pr4)
        #out_scale.insert(0, 4)
        pr4_up = self.upsample(pr4)

        deconv3 = self.deconv3(conv4b)
        iconv3 = self.iconv3(torch.cat([deconv3, pr4_up, conv3b], dim = 1))
        pr3 = self.pr3(iconv3)
        #out.insert(0, pr3)
        #out_scale.insert(0, 3)
        pr3_up = self.upsample(pr3)

        deconv2 = self.deconv2(iconv3)
        iconv2 = self.iconv2(torch.cat([deconv2, pr3_up, conv2], dim = 1))
        pr2 = self.pr2(iconv2)
        #out.insert(0, pr2)
        #out_scale.insert(0, 2)
        pr2_up = self.upsample(pr2)

        deconv1 = self.deconv1(iconv2)
        iconv1 = self.iconv1(torch.cat([deconv1, pr2_up, conv1], dim = 1))
        pr1 = self.pr1(iconv1)
        #out.insert(0, pr1)
        #out_scale.insert(0, 1)
        pr1_up = self.upsample(pr1)

        #pr0 = pr1[:, :, :imL.shape[-2], :imL.shape[-1]]
        # out.insert(0, pr0)
        # out_scale.insert(0, 0)
        if(mode == "test"): out[-1] = out[-1].clamp(self.delt, maxD)

        # return out_scale, out
        # return [pr1_up, pr1, pr2, pr3, pr4]
        return pr1_up[:, :, :imL.shape[2]-pad_numbers[3], :imR.shape[3]-pad_numbers[1]]


class Dispnet_2to2(nn.Module):
    def __init__(self, maxdisparity=192):
        super(Dispnet_2to2, self).__init__()
        self.name = "Dispnet_v2"
        self.D = maxdisparity
        self.delt = 1e-6
        self.count_levels = 7

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = conv2d_bn(2, 64, kernel_size=7, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv2 = conv2d_bn(64, 128, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3a = conv2d_bn(128, 256, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3b = conv2d_bn(256, 256, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv4a = conv2d_bn(256, 512, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv4b = conv2d_bn(512, 512, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)

        self.pr4 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

        self.deconv3 = deconv2d_bn(512, 128, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv3 = conv2d_bn(385, 128, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.deconv2 = deconv2d_bn(128, 64, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv2 = conv2d_bn(193, 64, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.deconv1 = deconv2d_bn(64, 32, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv1 = conv2d_bn(97, 32, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr1 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)

        net_init(self)
        for m in [self.pr4, self.pr3, self.pr2, self.pr1]:
            m.weight.data = m.weight.data*0.1

    def forward(self, imL, imR, mode="train"):
        assert imL.shape == imR.shape
        maxD = max(self.D, imL.shape[-1])
        out = []

        pad_numbers = (0, ceil(imL.shape[3] / 16.) * 16 - imL.shape[3], 0, ceil(imL.shape[2] / 16.) * 16 - imL.shape[2])

        imL = nn.functional.pad(imL, pad_numbers)
        imR = nn.functional.pad(imR, pad_numbers)

        x = torch.cat([imL, imR], dim=1)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3a = self.conv3a(conv2)
        conv3b = self.conv3b(conv3a)
        conv4a = self.conv4a(conv3b)
        conv4b = self.conv4b(conv4a)

        pr4 = self.pr4(conv4b)
        #out.insert(0, pr4)
        #out_scale.insert(0, 4)
        pr4_up = self.upsample(pr4)

        deconv3 = self.deconv3(conv4b)
        iconv3 = self.iconv3(torch.cat([deconv3, pr4_up, conv3b], dim = 1))
        pr3 = self.pr3(iconv3)
        #out.insert(0, pr3)
        #out_scale.insert(0, 3)
        pr3_up = self.upsample(pr3)

        deconv2 = self.deconv2(iconv3)
        iconv2 = self.iconv2(torch.cat([deconv2, pr3_up, conv2], dim = 1))
        pr2 = self.pr2(iconv2)
        #out.insert(0, pr2)
        #out_scale.insert(0, 2)
        pr2_up = self.upsample(pr2)

        deconv1 = self.deconv1(iconv2)
        iconv1 = self.iconv1(torch.cat([deconv1, pr2_up, conv1], dim = 1))
        pr1 = self.pr1(iconv1)
        #out.insert(0, pr1)
        #out_scale.insert(0, 1)
        pr1_up = self.upsample(pr1)

        #pr0 = pr1[:, :, :imL.shape[-2], :imL.shape[-1]]
        # out.insert(0, pr0)
        # out_scale.insert(0, 0)
        if(mode == "test"): out[-1] = out[-1].clamp(self.delt, maxD)

        # return out_scale, out
        # return [pr1_up, pr1, pr2, pr3, pr4]
        return pr1_up[:, :, :imL.shape[2]-pad_numbers[3], :imR.shape[3]-pad_numbers[1]]

