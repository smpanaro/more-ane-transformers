#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
import torch.nn as nn

class KahanLayerNormANE(nn.Module):
    """
    LayerNorm optimized for Apple Neural Engine (ANE) execution, using Kahan summation
    to attempt to reduce numerical error. Unfortunately in practice, it slightly increases it.

    Note: This layer only supports normalization over the final dim. It expects `num_channels`
    as an argument and not `normalized_shape` which is used by `torch.nn.LayerNorm`.
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

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def check_overunderflow(self, x):
        f16_min = torch.finfo(torch.float16).min
        f16_max = torch.finfo(torch.float16).max

        # Check for values that are outside the range of float16
        ouf = torch.any(x < f16_min) or torch.any(x > f16_max) or torch.isnan(x).any() or torch.isinf(x).any()
        if ouf:
            print("x is out of bounds", x)

    @staticmethod
    def kahan_mean(inputs, size: int = 4):
        assert inputs.size(1) % size == 0, "Batch size must be divisible by channels size."
        s = torch.zeros((1,1,1,inputs.size(-1)), dtype=inputs.dtype, device=inputs.device) # total
        c = torch.zeros((1,1,1,inputs.size(-1)), dtype=inputs.dtype, device=inputs.device) # compensation
        rc = torch.zeros((1,1,1,inputs.size(-1)), dtype=inputs.dtype, device=inputs.device) # compensation

        # inputs = torch.sort(inputs, dim=1).values
        for batch in inputs.chunk(size, dim=1):
            # print("\n----\nbatch", batch)
            # print("batch_sum:", batch.sum(dim=1, keepdims=True))
            # print("c:", c)
            batch_sum = batch.sum(dim=1, keepdims=True)
            # y = batch_sum - c
            t = batch_sum

            # KBN sum
            abss = torch.abs(s)
            abst = torch.abs(batch_sum)
            max_st = torch.where(abss >= abst, s, batch_sum)
            min_st = torch.where(abss < abst, s, batch_sum)
            # print(max_st, min_st)
            c += (max_st - batch_sum) + min_st

            # print("y = batch_sum - c\n", y)
            # t = s + y
            # print("t = s + y\n", t)
            # print(c.shape, t.shape, s.shape, y.shape)
            # print(c.dtype, t.dtype, s.dtype, y.dtype)
            # c = (t - s) - y
            # rc += c
            # print("c = (t-s) - y\n", c)
            s = t
            # print("s = t\n", s)

            # batch_sum = batch.sum(dim=1, keepdims=True)
            # print("batch")
            # print(batch)
            # print("batch_sum")
            # print(batch_sum)
            # batch_comp = batch_sum - torch.sum(batch - c)
            # s += batch_sum
            # c = batch_comp
        # print("c", c)

        # print(s, rc)
        # print("torch.mean", inputs.mean(dim=1, keepdims=True))
        # print("kahan mean", s / inputs.size(1))
        return (s / inputs.size(1)) + (c / inputs.size(1))

    @staticmethod
    def stable_mean(inputs, size: int = 4):
        assert inputs.size(1) % size == 0, "Batch size must be divisible by channels size."
        # (x+y+z) / 3 = (x+y)/3 + z/3
        m = torch.zeros((1,1,1,inputs.size(-1)), dtype=inputs.dtype, device=inputs.device) # total
        for batch in inputs.chunk(size, dim=1):
            m += batch.sum(dim=1, keepdims=True) / inputs.size(1)

        # print("torch.mean", inputs.mean(dim=1, keepdims=True))
        # print("stable mean", m)
        return m


    def forward(self, inputs):
        input_rank = len(inputs.size())

        # Principle 1: Picking the Right Data Format (machinelearning.apple.com/research/apple-neural-engine)
        # Migrate the data format from BSC to BC1S (most conducive to ANE)
        if input_rank == 3 and inputs.size(2) == self.num_channels:
            inputs = inputs.transpose(1, 2).unsqueeze(2)
            input_rank = len(inputs.size())

        assert input_rank == self.expected_rank
        assert inputs.size(1) == self.num_channels
        assert inputs.dtype == torch.float16 or inputs.dtype == torch.float32

        if self.clip_mag is not None:
            inputs.clamp_(-self.clip_mag, self.clip_mag)

        # Reference implementation.
        # channels_mean = inputs.mean(dim=1, keepdims=True)

        # zero_mean = inputs - channels_mean

        # zero_mean_sq = zero_mean * zero_mean

        # denom = (zero_mean_sq.mean(dim=1, keepdims=True) + self.eps).rsqrt()

        # out = zero_mean * denom

        # Kahan implementation.
        size = 4
        channels_mean = self.kahan_mean(inputs, size=size)
        # channels_mean = self.stable_mean(inputs, size=size)

        zero_mean = inputs - channels_mean

        zero_mean_sq = zero_mean * zero_mean

        denom = (self.kahan_mean(zero_mean_sq, size=size) + self.eps).rsqrt()
        # denom = (self.stable_mean(zero_mean_sq, size=size) + self.eps).rsqrt()
        # denom = (zero_mean_sq.mean(dim=1, keepdims=True) + self.eps).rsqrt()

        out = zero_mean * denom

        if self.elementwise_affine:
            out = (out + self.bias.view(1, self.num_channels, 1, 1)
                   ) * self.weight.view(1, self.num_channels, 1, 1)
        # Deviate from ANE implementation, match nn.
        # if self.elementwise_affine:
        #     out = (out * self.weight.view(1, self.num_channels, 1, 1)
        #            ) + self.bias.view(1, self.num_channels, 1, 1)

        return out