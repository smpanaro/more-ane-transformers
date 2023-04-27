import torch
from torch import nn
import numpy as np
from src.ml_ane_transformers.ane.dummy_layer_norm import DummyLayerNormANE as DummyLN
from src.ml_ane_transformers.ane.layer_norm import LayerNormANE
import coremltools as ct
from src.utils.psnr import compute_psnr
from coremltools.converters.mil import Builder as mb
import sys

"""
Test out the DummyLayerNormANE and see if it's possible to hot swap
by overriding the batch_norm op.

It is.
"""

torch.manual_seed(42)

from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY
if "batch_norm" in _TORCH_OPS_REGISTRY:
    del _TORCH_OPS_REGISTRY["batch_norm"]

@register_torch_op
def batch_norm(context, node):
    inputs = _get_inputs(context, node, expected=9)
    _input = inputs[0]
    weight = inputs[1]
    bias = inputs[2]
    running_mean = inputs[3]
    running_var = inputs[4]
    training = inputs[5].val
    eps = inputs[7]

    context.add(eps)
    context.add(weight)
    context.add(bias)
    context.add(_input)
    print("weight.shape", weight.shape)

    ln = mb.layer_norm(x=_input, axes=[1], epsilon=eps, gamma=weight, beta=bias, name=node.name)
    context.add(ln)


B,C,S = 1, 5, 3
dummy = DummyLN(C)
ln = nn.LayerNorm(C, elementwise_affine=False)
ane = LayerNormANE(C, elementwise_affine=False)

dummy_trace = torch.jit.trace(dummy, torch.randn((B,C,1,S)))
prog = ct.convert(dummy_trace,
                inputs=[ct.TensorType("x", shape=(B,C,1,S))],
                outputs=[ct.TensorType("y")],
                # pass_pipeline=ct.PassPipeline(pipeline_name="empty"),
                compute_precision=ct.precision.FLOAT32, # The float16 loss is real.
                convert_to="milinternal")
print(prog)

mlmodel = ct.convert(prog,
                inputs=[ct.TensorType("x", shape=(B,C,1,S))],
                outputs=[ct.TensorType("y")],
                compute_precision=ct.precision.FLOAT32,
                convert_to="mlprogram")

x = torch.randn((B,C,1,S))
print("x", x)
print("y", mlmodel.predict({"x": x.numpy()})["y"])
print("ane", ane(x))
print("ln", ane(x.permute(0,3,1,2).squeeze(-1)))
