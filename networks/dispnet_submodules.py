import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import torch


flag_bias_t = True
flag_bn = False
activefun_t = nn.ReLU(inplace=True)
flag_check_shape = False
flag_bias_default = True
activefun_default = nn.ReLU(inplace=True)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).normal_(0.0, v)


def net_init(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.weight.data = fanin_init(m.weight.data.size())
        elif isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class conv_block(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU(inplace=True), dilation=1, is_BN=False, is_Pooling=False, group = 1):
        super(conv_block, self).__init__()
        #group = group
        block = [("conv", nn.Conv2d(inc, outc, kernel_size, padding=padding, dilation=dilation, stride=stride, bias=use_bias, groups = group))]

        if is_BN:
            block.append(("bn", nn.InstanceNorm2d(outc, affine=False)))
        if activation is not None:
            block.append(("act", activation))
        if is_Pooling:
            block.append(("maxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)))

        self.conv = nn.Sequential(OrderedDict(block))

    def forward(self, input):
        return self.conv(input)


class transpose_conv_block(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU(inplace=True), is_BN=False, is_Pooling=False):
        super(transpose_conv_block, self).__init__()
        if is_BN:
            self.conv = nn.Sequential(OrderedDict([
                ("conv", nn.ConvTranspose2d(inc, outc, kernel_size, padding=padding, stride=stride, bias=use_bias)),
                ("bn", nn.InstanceNorm2d(outc, affine=False)),
                ("act", activation)
            ]))
        elif is_Pooling:
            self.conv = nn.Sequential(OrderedDict([
                ("conv", nn.ConvTranspose2d(inc, outc, kernel_size, padding=padding, stride=stride, bias=use_bias)),
                ("act", activation),
                ("maxPool", nn.MaxPool2d(kernel_size=3, stride=2, padding=padding))
            ]))
        elif activation is not None:
            self.conv = nn.Sequential(OrderedDict([
                ("conv", nn.ConvTranspose2d(inc, outc, kernel_size, padding=padding, stride=stride, bias=use_bias)),
                ("act", activation)
            ]))
        else:
            self.conv = nn.Sequential(OrderedDict([
                ("conv", nn.ConvTranspose2d(inc, outc, kernel_size, padding=padding, stride=stride, bias=use_bias)),
            ]))
    def forward(self, input):
        return self.conv(input)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def make_layer_CARB(block, n_layers,nf,nframes):
    layers = []
    for _ in range(n_layers):
        layers.append(block(nf=nf, nframes=nframes))
    return nn.Sequential(*layers)


def initialize_weights(net_l, scale=1.):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)


class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class GatedDeConv2dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True,activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv2dWithActivation, self).__init__()
        self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        #print(input.size())
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)


class SNGatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convolution with spetral normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNGatedConv2dWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation
        self.batch_norm = batch_norm
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.mask_conv2d = torch.nn.utils.spectral_norm(self.mask_conv2d)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        return self.sigmoid(mask)
        #return torch.clamp(mask, -1, 1)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class SNGatedDeConv2dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNGatedDeConv2dWithActivation, self).__init__()
        self.conv2d = SNGatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        #print(input.size())
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)


class SNConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConvWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


def conv2d_bn(in_planes, out_planes, kernel_size=3, stride=1, flag_bias=flag_bias_default, bn=flag_bn, activefun=activefun_default):
    "2d convolution with padding, bn and activefun"
    assert kernel_size % 2 == 1
    conv2d = Conv2d(in_planes, out_planes, kernel_size, stride, padding=(kernel_size - 1)//2, bias=flag_bias)

    if(not bn and not activefun):
        return conv2d

    layers = []
    layers.append(conv2d)
    if bn: layers.append(nn.BatchNorm2d(out_planes))
    if activefun: layers.append(activefun)

    return nn.Sequential(*layers)


def deconv2d_bn(in_planes, out_planes, kernel_size=4, stride=2, flag_bias=flag_bias_default, bn=flag_bn, activefun=activefun_default):
    "2d deconvolution with padding, bn and activefun"
    assert stride > 1
    p = (kernel_size - 1)//2
    op = stride - (kernel_size - 2*p)
    conv2d = ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding=p, output_padding=op, bias=flag_bias)

    if(not bn and not activefun):
        return conv2d

    layers = []
    layers.append(conv2d)
    if bn: layers.append(nn.BatchNorm2d(out_planes))
    if activefun: layers.append(activefun)

    return nn.Sequential(*layers)


def myCat2d(*seq):
    assert len(seq[0].shape) == 4
    bn, c, h, w = seq[0].shape
    for tmp in seq:
        _, _, ht, wt = tmp.shape
        if(h > ht): h = ht
        if(w > wt): w = wt
    seq1 = [ seq[i][:, :, :h, :w] for i in range(len(seq))]
    return torch.cat(seq1, dim = 1)


class Conv2d(nn.Conv2d):
    def forward(self, obj_in):
        obj_out = super(Conv2d, self).forward(obj_in)
        return obj_out


class ConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, obj_in):
        obj_out = super(ConvTranspose2d, self).forward(obj_in)
        return obj_out
