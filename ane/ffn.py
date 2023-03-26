#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
import torch.nn as nn
import math

from .layer_norm import LayerNormANE

def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class FFN(nn.Module):

    def __init__(self, embed_dim, ffn_dim, dropout=0.1, **kwargs):
        super().__init__()
        self.c_fc = nn.Conv2d(embed_dim, ffn_dim, 1)
        self.c_proj = nn.Conv2d(ffn_dim, embed_dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def _forward_impl(self, x, **kwargs):
        # for l in self.layers:
        #     x = l(x)
        # return x
        # Implement manually?,
        x = self.c_fc(x)
        # Match OpenAI GPT implementation, gelu instead of relu.
        x = new_gelu(x)
        # x = self.relu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResidualFFN(FFN):

    def __init__(self, embed_dim, dropout=0.1, drop_fn=nn.Dropout, **kwargs):
        super().__init__(embed_dim, dropout=dropout, **kwargs)
        self.rdropout = drop_fn(dropout) if dropout > 0. else nn.Identity()
        self.rnorm = LayerNormANE(embed_dim)

    def forward(self, x):
        residual = self._forward_impl(x)
        return self.rnorm(self.rdropout(residual) + x)


class PreNormResidualFFN(ResidualFFN):

    def forward(self, x):
        residual = self.rdropout(self._forward_impl(self.rnorm(x)))
        return x + residual