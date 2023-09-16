import torch
from torch import nn
import numpy as np
import coremltools as ct
import time

"""
Experiment to see if it is possible to pass an MxN input, where M is enumerated and N is fixed,
and extract a fixed size KxN matrix as well as a variable size JxN matrix.

If this works, it can be used to pass a KV cache plus the input embeddings allowing enumerated
input IDs + KV caching + Neural Engine.

This is a cleaned up version of the enuemrated_shapes_transformer experiment.
"""

from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb

# del _TORCH_OPS_REGISTRY["constantchunk"]

# This is the magic (and sketchy) bit.
# Basically avoids the safety checks that are in place and we
# make sure not to pass anything in that doesn't divide evenly.
# @register_torch_op
# def constantchunk(context, node):
#     inputs = _get_inputs(context, node, expected=1)
#     x = inputs[0]
#     # ConstantChunk gets its parameters as attributes of the node.
#     chunks = node.attr["chunks"]
#     dim = node.attr["dim"]

#     res = mb.split(x=x, num_splits=chunks, axis=dim, name=node.name)
#     for val, name in zip(res, node.outputs):
#         context.add(val, name)

class Net(nn.Module):
    def __init__(self, n, k):
        self.n = n
        self.k = k

    def forward(self, x):
        # x is shape (M,N)
        assert x.shape[1] == self.n

        # MxN matrix, first K rows are a KxN matrix, the rest are a JxN matrix.
        # M1, K1 [1, 2, ..., N]
        # M2, K2 [1, 2, ..., N]
        # M3, K3 [1, 2, ..., N]
        # ...
        # Mx, J1 [1, 2, ..., N]
        # My, J2 [1, 2, ..., N]
        # ...

        ks,js  = x[:self.k, :], x[self.k:, :]





