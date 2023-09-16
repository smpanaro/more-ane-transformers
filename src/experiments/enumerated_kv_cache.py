import coremltools as ct
import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from os_signpost import Signposter
import os
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

"""
Mini model to test the idea of a fixed length input_ids, but
enumerated shapes for the KV cache. Make sure it runs on the ANE.

Started 9/10/2023

Does not run on ANE sadly. Maybe because multiple symbols get introduced
even though there really is only one (the splitting is symbol+constant-constant).
"""

B,C,S = 1,768,128 # batch size, channels, sequence length
# K - cache size, enumerated
V = 10_000 # vocab size, under 16k for ANE

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, C)

    def forward(self, x, cache):
        # x: [B,S]
        # cache: [B,C,1,K]
        # values are directionally representative of single head attention but not exactly

        x = self.emb(x) # [B,S,C]
        x = x.transpose(1,2) # [B,C,S]
        x = x.unsqueeze(-2) # [B,C,1,S]

        # tensor<fp16, [1, 64, 1, 136]> var_589_cast = concat(axis = var_318, interleave = var_589_interleave_0, values = (var_531_cast_0, var_495_cast_0)); tensor<int32, [4]> var_593_begin_0 = const()[name = tensor<string, []>("op_593_begin_0"), val = tensor<int32, [4]>([0, 0, 0, 8])];
        # tensor<int32, [4]> var_593_end_0 = const()[name = tensor<string, []>("op_593_end_0"), val = tensor<int32, [4]>([1, 64, 1, 136])];
        # tensor<bool, [4]> var_593_end_mask_0 = const()[name = tensor<string, []>("op_593_end_mask_0"), val = tensor<bool, [4]>([true, true, true, true])];
        # tensor<fp16, [1, 64, 1, 128]> var_593_cast = slice_by_index(begin = var_593_begin_0, end = var_593_end_0, end_mask = var_593_end_mask_0, x = var_589_cast);

        # this should be k, q should be roughly x
        # this doesn't work with enumerated shapes
        q = torch.cat([cache, x], dim=-1) # [B,C,1,S+K]
        q = q[...,S:] # [B,C,1,K]
        return q
        # q = q.reshape(B,1,-1,C) # [B,1,K,C]
        # kT = q.transpose(-1,-2) # for simplicity reuse same matrix, [B,1,C,K]
        # return q@kT

net = Net()
net.eval()
trace_input = (torch.randint(0, V, (B,S)), torch.randn(B,C,1,512))
print(net(*trace_input))
print([x.size() for x in net(*trace_input)])
traced = torch.jit.trace(net, trace_input)

cache_shapes = ct.EnumeratedShapes(shapes=[
    [B,C,1,128],
    [B,C,1,256],
    [B,C,1,512],
    [B,C,1,1024],
], default=[B,C,1,512])
cache_default_shape = ct.Shape(shape=[B,C,1,512])

convert_args = {
    "inputs": [
        ct.TensorType("x", shape=ct.Shape(shape=[1,S]), dtype=np.float16),
        ct.TensorType("cache", shape=cache_shapes, dtype=np.float16),
    ],
    "outputs": [
        ct.TensorType("out", dtype=np.float16),
    ],
    "compute_precision": ct.precision.FLOAT16,
    "minimum_deployment_target": ct.target.iOS16,
    "compute_units": ct.ComputeUnit.CPU_AND_NE,
}
mlprog = ct.convert(
    traced,
    convert_to="milinternal",
    **convert_args,
)
print("mlprog:")
print(mlprog)

model = ct.convert(
    mlprog,
    convert_to="mlprogram",
    **convert_args,
)

model.save("enumerated-kv-cache.mlpackage")

print("PID:", os.getpid())
input("prese enter to go")
print()

signposter = Signposter("com.example.my_subsystem", Signposter.Category.PointsOfInterest)

x = torch.randint(0, V, (1,S)).int()
default_cache = torch.randn(B,C,1,512).float()
other_cache = torch.randn(B,C,1,256).float()

print(f"predicting default")
end_interval = signposter.begin_interval("default shape")
print(model.predict({"x": x, "cache": default_cache}))
end_interval("end default shape")

# print(f"predicting other")
# end_interval = signposter.begin_interval("other shape")
# print(model.predict({"x": x, "cache": other_cache}))
# end_interval("end other shape")
