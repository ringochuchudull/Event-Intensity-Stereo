import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    def __init__(self, kernel_size, n_feat, left_nc=1, nhidden=64):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(n_feat, affine=False)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(left_nc, nhidden, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, n_feat, kernel_size=kernel_size, padding=kernel_size // 2)
        self.mlp_beta = nn.Conv2d(nhidden, n_feat, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x, left):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        left_map = F.interpolate(left, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(left_map)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, left_nc=1,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(SPADEResBlock, self).__init__()


        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.act1 = nn.ReLU()
        self.norm = SPADE(kernel_size, n_feat, left_nc=left_nc, nhidden=int(n_feat / 4))
        self.conv2 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.act2 = nn.ReLU()
        self.res_scale = res_scale

    def forward(self, x, left):
        res = self.act1(self.conv1(x))
        res = self.act2(self.conv2(self.norm(res, left)))
        res = res.mul(self.res_scale)
        res += x

        return res


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def warping_events(event_bins, flow):
    """
    :param events: a batch of event bins (B, 2, num_bins, H, W)
    :param flow: two-channel tensor, indicating x and y. (B, 2, H, W)
    :return: a two-channel tensor that present warped positive and negative events (B, 2, H, W)
    """
    B, _, num_bins, h, w = event_bins.shape
    rtime_of_bins = (torch.arange(num_bins * 1.0) / num_bins).expand([B, num_bins]).reshape((B * num_bins, 1, 1, 1)).type_as(event_bins)  # (B * num_bins, 1, 1, 1)

    event_bins_batch = event_bins.permute(0, 2, 1, 3, 4).reshape((B * num_bins, 2, h, w))  # (B * num_bins, 2, h, w)

    xx = torch.arange(0, w).view(1, -1).repeat(h, 1).view(1, 1, h, w).repeat(B * num_bins, 1, 1, 1)
    yy = torch.arange(0, h).view(-1, 1).repeat(1, w).view(1, 1, h, w).repeat(B * num_bins, 1, 1, 1)
    base_grid = torch.cat((xx, yy), 1).float()

    flow_batch = rtime_of_bins * flow.unsqueeze(1).repeat(1, num_bins, 1, 1, 1).reshape((B * num_bins, 2, h, w))  # (B * num_bins, 2, h, w)
    flow_grid = torch.cat([flow_batch[:, 0:1, :, :], flow_batch[:, 1:2, :, :]], dim=1)
    gird = (flow_grid + base_grid.type_as(flow_grid)).permute(0, 2, 3, 1)
    gird[:, :, :, 0] = gird[:, :, :, 0] / ((w - 1.) / 2.) - 1.
    gird[:, :, :, 1] = gird[:, :, :, 1] / ((h - 1.) / 2.) - 1.

    warped_event_batch = F.grid_sample(event_bins_batch, gird,
                                       mode='bilinear', #mode='bilinear',  # mode='nearest'
                                       padding_mode='zeros')  # (B * num_bins, 2, h, w)
    # warped_event_bins = torch.sum(warped_event_batch.reshape((B, num_bins, 2, h, w)), 1)
    warped_event_bins = warped_event_batch.reshape((B, num_bins, 2, h, w))

    return warped_event_bins


def warping_disparity(img, disp):
    '''
    img.shape = b, c, h, w
    disp.shape = b, 1, h, w
    '''
    b, c, h, w = img.shape

    right_coor_x = torch.arange(0, w).repeat(b, 1, h, 1).type_as(disp)
    right_coor_y = torch.arange(0, h).repeat(b, 1, w, 1).transpose(2, 3).type_as(disp)
    left_coor_x1 = right_coor_x + disp
    left_coor_norm1 = torch.cat([left_coor_x1 / (w - 1) * 2 - 1, right_coor_y / (h - 1) * 2 - 1], dim=1)

    ## backward warp
    warp_img = torch.nn.functional.grid_sample(img, left_coor_norm1.permute(0, 2, 3, 1))
    return warp_img