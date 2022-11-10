import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.common import ResBlock, default_conv, Upsampler, SPADEResBlock
from models.losses import create_window

from argparse import ArgumentParser


class AdaptorBaselineOne2One(nn.Module):
    def __init__(self,
                 num_channels=1,
                 width=64,
                 depth=8):
        super(AdaptorBaselineOne2One, self).__init__()
        n_resblock = depth
        n_feats = width
        kernel_size = 3

        # define head module
        m_head = [nn.Conv2d(num_channels, n_feats, kernel_size=5, padding=2),
                  nn.Conv2d(n_feats, n_feats, kernel_size=5, padding=2, stride=2),
                  nn.Conv2d(n_feats, n_feats, kernel_size=5, padding=2, stride=2)]

        # define body module
        m_body = [
            ResBlock(
                default_conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=1.
            ) for _ in range(n_resblock)
        ]
        m_body.append(default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(default_conv, 4, n_feats, act=False),
            nn.Conv2d(num_channels, n_feats, kernel_size=5, padding=2)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_channels', type=int, default=1)
        parser.add_argument('--width', type=int, default=64)
        parser.add_argument('--depth', type=int, default=8)
        return parser

    @staticmethod
    def from_namespace(args):
        instance = AdaptorBaselineOne2One(
            num_channels=args.num_channels,
            width=args.width,
            depth=args.depth
        )
        return instance

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x


class AdaptorBaselineOne2OneWtLeft(nn.Module):
    def __init__(self,
                 num_channels=1,
                 width=64,
                 depth=8):
        super(AdaptorBaselineOne2OneWtLeft, self).__init__()
        n_resblock = depth
        self.n_resblock = depth
        n_feats = width
        kernel_size = 3

        # define head module
        m_head = [nn.Conv2d(num_channels, n_feats, kernel_size=5, padding=2),
                  nn.Conv2d(n_feats, n_feats, kernel_size=5, padding=2, stride=2),
                  nn.Conv2d(n_feats, n_feats, kernel_size=5, padding=2, stride=2)]

        # define body module
        for i in range(n_resblock):
            self.add_module(f'spade_resblock_{i}', SPADEResBlock(default_conv, n_feats, kernel_size, left_nc=num_channels))
        m_body = [
            default_conv(n_feats, n_feats, kernel_size)
        ]

        # define tail module
        m_tail = [
            Upsampler(default_conv, 4, n_feats, act=False),
            nn.Conv2d(n_feats, num_channels, kernel_size=5, padding=2)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_channels', type=int, default=1)
        parser.add_argument('--width', type=int, default=64)
        parser.add_argument('--depth', type=int, default=6)
        return parser

    @staticmethod
    def from_namespace(args):
        instance = AdaptorBaselineOne2OneWtLeft(
            num_channels=args.num_channels,
            width=args.width,
            depth=args.depth
        )
        return instance

    def forward(self, x, left):
        x = self.head(x)
        res = x
        left_map = F.interpolate(left, size=res.shape[-2:], mode='nearest')

        for i in range(self.n_resblock):
            res = self.__getattr__(f'spade_resblock_{i}')(res, left_map)

        res = self.body(res)
        res += x

        x = self.tail(res)

        return x


class AdaptorBaselineOne2OneWtLeftV2(nn.Module):
    def __init__(self,
                 num_channels=1,
                 width=64,
                 depth=8):
        super(AdaptorBaselineOne2OneWtLeftV2, self).__init__()
        n_resblock = depth
        self.n_resblock = depth
        n_feats = width
        kernel_size = 3
        self.window = create_window(21, 1, sigma=6)
        self.window_size = 21

        # define head module
        m_head = [nn.Conv2d(num_channels, n_feats, kernel_size=5, padding=2),
                  nn.Conv2d(n_feats, n_feats, kernel_size=5, padding=2, stride=2),
                  nn.Conv2d(n_feats, n_feats, kernel_size=5, padding=2, stride=2)]

        # define body module
        for i in range(n_resblock):
            self.add_module(f'spade_resblock_{i}', SPADEResBlock(default_conv, n_feats, kernel_size, left_nc=num_channels))
        m_body = [
            default_conv(n_feats, n_feats, kernel_size)
        ]

        # define tail module
        m_tail = [
            Upsampler(default_conv, 4, n_feats, act=False),
            nn.Conv2d(n_feats, num_channels, kernel_size=5, padding=2)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_channels', type=int, default=1)
        parser.add_argument('--width', type=int, default=64)
        parser.add_argument('--depth', type=int, default=6)
        return parser

    @staticmethod
    def from_namespace(args):
        instance = AdaptorBaselineOne2OneWtLeftV2(
            num_channels=args.num_channels,
            width=args.width,
            depth=args.depth
        )
        return instance

    def forward(self, x, left):
        x = self.head(x)
        res = x
        left_map = F.conv2d(F.interpolate(left, size=res.shape[-2:], mode='bilinear'), self.window.type_as(left), padding=self.window_size // 2, groups=left.shape[1])

        for i in range(self.n_resblock):
            res = self.__getattr__(f'spade_resblock_{i}')(res, left_map)

        res = self.body(res)
        res += x

        x = self.tail(res)

        return x

