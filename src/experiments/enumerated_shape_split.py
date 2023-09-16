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
See if we can pass in a single enumerated input
and split it into a fixed number of pieces using only one symbolic variable.

9/12/23 -- Seems like it introduces new symbolic variables.
"""

class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [m] where m == 2n or m == n+1
        # y,z = x.split(2)
        y = x[:,:x.shape[1]//2,:,:]
        z = x[:,x.shape[1]//2:,:,:]
        return y@z.transpose(2,3)

net = Net()
net.eval()
trace_input = (torch.randint(0, 100, (1, 128,1,512)),)
print(net(*trace_input))
print([x.size() for x in net(*trace_input)])
traced = torch.jit.trace(net, trace_input)


x_fixed_shape = ct.Shape(shape=[1,5120,1,2000])
x_flex_shape = ct.Shape(shape=(1,ct.RangeDim(lower_bound=1, upper_bound=10240, default=1),2000,1))
x_enum_shape = ct.EnumeratedShapes(shapes=[
    [1,2560,1,2000],
    # [1,2000,1,2561],
    [1,5120,1,2000],
    # [1,2000,1,5121],
    # [1,2000,1,10240],
    # [1,2000,1,10241],
], default=[1,5120,1,2000])

# shape = (1,ct.RangeDim().symbol,1,2000,)
# @mb.program(input_specs=[mb.TensorSpec(shape=shape, dtype=types.float),], opset_version=ct.target.iOS16)
# def prog(x):
#     # y,z = mb.split(x=x, num_splits=2, axis=1)
#     # y = mb.expand_dims(x=y, axes=0)
#     # z = mb.expand_dims(x=z, axes=1)
#     a = mb.matmul(x=y, y=z, transpose_y=True)
#     return a
#     # half = mb.const(val=0.05)
#     # third = mb.const(val=0.03)
#     # a = mb.mul(x=a, y=half)


#     # b = mb.mul(x=y, y=half)
#     # b = mb.matmul(x=b, y=z, transpose_y=True)

#     # c = mb.mul(x=z, y=half)
#     # c = mb.matmul(x=y, y=c, transpose_y=True)

#     # d = mb.mul(x=y, y=third)
#     # d = mb.matmul(x=d, y=z, transpose_y=True)

#     # e = mb.mul(x=z, y=third)
#     # e = mb.matmul(x=y, y=e, transpose_y=True)

#     # s = mb.add(x=a, y=b)
#     # s = mb.add(x=s, y=c)
#     # s = mb.add(x=s, y=d)
#     # s = mb.add(x=s, y=e)
#     # return s

# traced = prog

print(f"traced: {traced}")

convert_args = {
    "inputs": [
        ct.TensorType("x", shape=x_enum_shape, dtype=np.float16),
    ],
    "outputs": [
        ct.TensorType("y", dtype=np.float16),
        # ct.TensorType("z", dtype=np.float16),
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
    pass_pipeline=ct.PassPipeline.EMPTY,
    **convert_args,
)

print("PID:", os.getpid())
input("prese enter to go")
print()

signposter = Signposter("com.example.my_subsystem", Signposter.Category.PointsOfInterest)

print(f"predicting default")
end_interval = signposter.begin_interval("default shape")
print(model.predict({"x": torch.randint(0, 10, (1,5120,1,2000,)).float()}))
end_interval("end default shape")

print(f"predicting default")
end_interval = signposter.begin_interval("other shape")
print(model.predict({"x": torch.randint(0, 10, (1,2560,1,2000,)).float()}))
end_interval("end default shape")