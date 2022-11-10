import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser


class CMNet3DConvD16S8(nn.Module):
    """
    This Network input (B, 2, 16, H, W) output (B, 2, H, W)
    Down-sample in D is 16, spatial 4
    """
    def __init__(self, num_channels=32):
        super(CMNet3DConvD16S8, self).__init__()

        self.conv3d_1 = nn.Conv3d(2, num_channels, kernel_size=5, stride=[2, 2, 2], padding=2)
        self.lrelu_1 = nn.LeakyReLU(0.1)
        self.conv3d_2 = nn.Conv3d(num_channels, num_channels, kernel_size=5, stride=[2, 2, 2], padding=2)
        self.lrelu_2 = nn.LeakyReLU(0.1)
        self.conv3d_3 = nn.Conv3d(num_channels, num_channels, kernel_size=5, stride=[2, 2, 2], padding=2)
        self.lrelu_3 = nn.LeakyReLU(0.1)
        self.conv3d_4 = nn.Conv3d(num_channels, num_channels, kernel_size=5, stride=[2, 1, 1], padding=2)
        self.lrelu_4 = nn.LeakyReLU(0.1)

        self.final_conv = nn.Conv2d(num_channels, 2, kernel_size=5, padding=2)

    def forward(self, event_bins):
        """
        :param event_bins: (B, 2, num_bins, max_h, max_w) num_bins should be 16
        :return:
        """
        B, _, __, h, w = event_bins.size()
        mid_3dbins_1 = self.lrelu_1(self.conv3d_1(event_bins))  # (B, num_channels, num_bins/2, max_h/2, max_w/2)
        mid_3dbins_2 = self.lrelu_1(self.conv3d_2(mid_3dbins_1))  # (B, num_channels, num_bins/4, max_h/2, max_w/2)
        mid_3dbins_3 = self.lrelu_1(self.conv3d_3(mid_3dbins_2))  # (B, num_channels, num_bins/8, max_h/4, max_w/4)
        mid_3dbins_4 = self.lrelu_1(self.conv3d_4(mid_3dbins_3))  # (B, num_channels, num_bins/16, max_h/4, max_w/4)
        mid_2dbins = F.interpolate(mid_3dbins_4.squeeze(2), size=(h, w), mode='bilinear')
        return self.final_conv(mid_2dbins)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_channels', type=int, default=32)

        return parser

    @staticmethod
    def from_namespace(args):
        instance = CMNet3DConvD16S8(
            num_channels=args.num_channels
        )
        return instance


class CMNet3DConvAffine(nn.Module):
    """
    This Network input (B, 2, 16, H, W) output (B, 2, H, W)
    Down-sample in D is 16, spatial 4
    """
    def __init__(self, num_channels=32):
        super(CMNet3DConvAffine, self).__init__()

        self.conv3d_1 = nn.Conv3d(2, num_channels, kernel_size=3, stride=[2, 2, 2], padding=1)
        self.lrelu_1 = nn.LeakyReLU(0.1)
        self.conv3d_2 = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=[2, 2, 2], padding=1)
        self.lrelu_2 = nn.LeakyReLU(0.1)
        self.conv3d_3 = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=[2, 2, 2], padding=1)
        self.lrelu_3 = nn.LeakyReLU(0.1)
        self.conv3d_4 = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=[2, 1, 1], padding=1)
        self.lrelu_4 = nn.LeakyReLU(0.1)

    def forward(self, event_bins):
        """
        :param event_bins: (B, 2, num_bins, max_h, max_w) num_bins should be 16
        :return:
        """
        B, _, __, h, w = event_bins.size()
        mid_3dbins_1 = self.lrelu_1(self.conv3d_1(event_bins))  # (B, num_channels, num_bins/2, max_h/2, max_w/2)
        mid_3dbins_2 = self.lrelu_1(self.conv3d_2(mid_3dbins_1))  # (B, num_channels, num_bins/4, max_h/2, max_w/2)
        mid_3dbins_3 = self.lrelu_1(self.conv3d_3(mid_3dbins_2))  # (B, num_channels, num_bins/8, max_h/4, max_w/4)
        mid_3dbins_4 = self.lrelu_1(self.conv3d_4(mid_3dbins_3))  # (B, num_channels, num_bins/16, max_h/4, max_w/4)
        mid_2dbins = F.interpolate(mid_3dbins_4.squeeze(2), size=(h, w), mode='bilinear')
        return self.final_conv(mid_2dbins)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_channels', type=int, default=32)

        return parser

    @staticmethod
    def from_namespace(args):
        instance = CMNet3DConvD16S8(
            num_channels=args.num_channels
        )
        return instance


