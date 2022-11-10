import torch
import torch.nn as nn
import numpy as np
from .firenet_unet import UNet, UNetRecurrent, UNetFire


class E2VIDRecurrent(nn.Module):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self,
                 num_bins=5,
                 skip_type='sum',
                 num_encoders=3,
                 base_num_channels=32,
                 num_residual_blocks=2,
                 norm='BN',
                 use_upsample_conv=False,
                 recurrent_block_type='convlstm'):
        super(E2VIDRecurrent, self).__init__()
        self.num_bins=num_bins,
        self.skip_type=skip_type,
        self.num_encoders=num_encoders,
        self.base_num_channels=base_num_channels,
        self.num_residual_blocks=num_residual_blocks,
        self.norm=norm,
        self.use_upsample_conv=use_upsample_conv,
        self.recurrent_block_type=recurrent_block_type

        self.unetrecurrent = UNetRecurrent(num_input_channels=self.num_bins,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """
        img_pred, states = self.unetrecurrent.forward(event_tensor, prev_states)
        return img_pred, states


