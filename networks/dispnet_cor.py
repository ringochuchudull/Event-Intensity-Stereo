import torch.nn as nn
from networks.dispnet_cor_submodules import *
import torch
from math import ceil

flag_bias_t = True
flag_bn = False
activefun_t = nn.ReLU(inplace=True)

class DispnetCorDual(nn.Module):
    def __init__(self, input_dim, maxdisparity=192):
        super(DispnetCorDual, self).__init__()
        self.name = "DispnetCorDual"
        self.D = maxdisparity
        self.delt = 1e-6
        self.count_levels = 7

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = conv2d_bn(input_dim, 64, kernel_size=7, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv2 = conv2d_bn(64, 128, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.corr = Corr1d(kernel_size=1, stride=1, D=41, simfun=None)
        self.redir = conv2d_bn(128, 64, kernel_size=1, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3a = conv2d_bn(64+41, 256, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
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

        pad_numbers = (0, ceil(imL.shape[3] / 16.) * 16 - imL.shape[3], 0, ceil(imL.shape[2] / 16.) * 16 - imL.shape[2])

        imL = nn.functional.pad(imL, pad_numbers)
        imR = nn.functional.pad(imR, pad_numbers)

        #Feature pyramid extractor
        conv1L = self.conv1(imL)
        conv1R = self.conv1(imR)
        conv2L = self.conv2(conv1L)
        conv2R = self.conv2(conv1R)

        #corr for cala left disparity, warp right view to left
        corr_l = self.corr(conv2L, conv2R)
        redir_l = self.redir(conv2L)
        conv3a_l = self.conv3a(torch.cat([corr_l, redir_l], dim=1))
        conv3b_l = self.conv3b(conv3a_l)
        conv4a_l = self.conv4a(conv3b_l)
        conv4b_l = self.conv4b(conv4a_l)

        pr4_l = self.pr4(conv4b_l)
        pr4_up_l = self.upsample(pr4_l)

        deconv3_l = self.deconv3(conv4b_l)
        iconv3_l = self.iconv3(torch.cat([deconv3_l, pr4_up_l, conv3b_l], dim = 1))
        pr3_l = self.pr3(iconv3_l)
        pr3_up_l = self.upsample(pr3_l)

        deconv2_l = self.deconv2(iconv3_l)
        iconv2_l = self.iconv2(torch.cat([deconv2_l, pr3_up_l, conv2L], dim = 1))
        pr2_l = self.pr2(iconv2_l)
        pr2_up_l = self.upsample(pr2_l)

        deconv1_l = self.deconv1(iconv2_l)
        iconv1_l = self.iconv1(torch.cat([deconv1_l, pr2_up_l, conv1L], dim = 1))
        pr1_l = self.pr1(iconv1_l)
        pr1_up_l = self.upsample(pr1_l)

        #corr for cala right disparity, warp left view to right
        corr_r = self.corr(conv2R, conv2L)
        redir_r = self.redir(conv2R)
        conv3a_r = self.conv3a(torch.cat([corr_r, redir_r], dim=1))
        conv3b_r = self.conv3b(conv3a_r)
        conv4a_r = self.conv4a(conv3b_r)
        conv4b_r = self.conv4b(conv4a_r)

        pr4_r = self.pr4(conv4b_r)
        pr4_up_r = self.upsample(pr4_r)

        deconv3_r = self.deconv3(conv4b_r)
        iconv3_r = self.iconv3(torch.cat([deconv3_r, pr4_up_r, conv3b_r], dim = 1))
        pr3_r = self.pr3(iconv3_r)
        pr3_up_r = self.upsample(pr3_r)

        deconv2_r = self.deconv2(iconv3_r)
        iconv2_r = self.iconv2(torch.cat([deconv2_r, pr3_up_r, conv2L], dim = 1))
        pr2_r = self.pr2(iconv2_r)
        pr2_up_r = self.upsample(pr2_r)

        deconv1_r = self.deconv1(iconv2_r)
        iconv1_r = self.iconv1(torch.cat([deconv1_r, pr2_up_r, conv1L], dim = 1))
        pr1_r = self.pr1(iconv1_r)
        pr1_up_r = self.upsample(pr1_r)

        # return [torch.cat((pr1_up_l, pr1_up_r), dim=1),
        #         torch.cat((pr1_l, pr1_r), dim=1),
        #         torch.cat((pr2_l, pr2_r), dim=1),
        #         torch.cat((pr3_l, pr3_r), dim=1),
        #         torch.cat((pr4_l, pr4_r), dim=1)]
        return torch.cat([pr1_up_l, pr1_up_r], dim=1)[:, :, :imL.shape[2]-pad_numbers[3], :imR.shape[3]-pad_numbers[1]]


        # return [pr1_up_l, pr1_l, pr2_l, pr3_l, pr4_l], [pr1_up_r, pr1_r, pr2_r, pr3_r, pr4_r]


class DispnetCor(nn.Module):
    def __init__(self, input_dim, max_disp=128):
        super(DispnetCor, self).__init__()
        self.name = "DispnetCor"
        self.D = max_disp
        self.delt = 1e-6
        self.count_levels = 7

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = conv2d_bn(input_dim, 64, kernel_size=7, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv2 = conv2d_bn(64, 128, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.corr = Corr1d(kernel_size=1, stride=1, D=41, simfun=None)
        self.redir = conv2d_bn(128, 64, kernel_size=1, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3a = conv2d_bn(64+41, 256, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
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

        pad_numbers = (0, ceil(imL.shape[3] / 16.) * 16 - imL.shape[3], 0, ceil(imL.shape[2] / 16.) * 16 - imL.shape[2])

        imL = nn.functional.pad(imL, pad_numbers)
        imR = nn.functional.pad(imR, pad_numbers)

        conv1L = self.conv1(imL)
        conv1R = self.conv1(imR)

        conv2L = self.conv2(conv1L)
        conv2R = self.conv2(conv1R)
        corr = self.corr(conv2L, conv2R)
        redir = self.redir(conv2L)
        conv3a = self.conv3a(torch.cat([corr, redir], dim=1))
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
        iconv2 = self.iconv2(torch.cat([deconv2, pr3_up, conv2L], dim = 1))
        pr2 = self.pr2(iconv2)
        #out.insert(0, pr2)
        #out_scale.insert(0, 2)
        pr2_up = self.upsample(pr2)

        deconv1 = self.deconv1(iconv2)
        iconv1 = self.iconv1(torch.cat([deconv1, pr2_up, conv1L], dim = 1))
        pr1 = self.pr1(iconv1)
        #out.insert(0, pr1)
        #out_scale.insert(0, 1)
        pr1_up = self.upsample(pr1)

        #pr0 = pr1[:, :, :imL.shape[-2], :imL.shape[-1]]
        #out.insert(0, pr0)
        #out_scale.insert(0, 0)
        if(mode == "test"): out[-1] = out[-1].clamp(self.delt, maxD)

        #return out_scale, out
        # return [pr1_up, pr1, pr2, pr3, pr4]
        return pr1_up[:, :, :imL.shape[2]-pad_numbers[3], :imR.shape[3]-pad_numbers[1]]