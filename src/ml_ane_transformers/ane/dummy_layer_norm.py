#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
import torch.nn as nn

class DummyLayerNormANE(nn.Module):
    """
    This looks like the ANELayerNorm class, but it just calls through to the
    torch batch norm (mostly same inputs/outputs as layer norm) and we replace
    that with a custom coremltools mb (torch layer norm doesn't support
    the 1st dimension, MIL does).
    """

    def __init__(self,
                 num_channels,
                 clip_mag=None,
                 eps=1e-5,
                 elementwise_affine=True):
        """
        Args:
            num_channels:       Number of channels (C) where the expected input data format is BC1S. S stands for sequence length.
            clip_mag:           Optional float value to use for clamping the input range before layer norm is applied.
                                If specified, helps reduce risk of overflow.
            eps:                Small value to avoid dividing by zero
            elementwise_affine: If true, adds learnable channel-wise shift (bias) and scale (weight) parameters
        """
        super().__init__()
        # Principle 1: Picking the Right Data Format (machinelearning.apple.com/research/apple-neural-engine)
        self.expected_rank = len('BC1S')

        self.num_channels = num_channels
        self.eps = eps
        self.clip_mag = clip_mag
        self.elementwise_affine = elementwise_affine

        self.bn = nn.BatchNorm2d(num_channels, eps=self.eps, affine=self.elementwise_affine)

    def forward(self, inputs):
        out = self.bn(inputs)
        return out