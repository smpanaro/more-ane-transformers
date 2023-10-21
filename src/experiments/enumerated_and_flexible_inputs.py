import coremltools as ct
import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from os_signpost import Signposter
import os

"""
See if it's possible to have a model that takes both an enumerated input
and a flexible shape input.

(I'm pretty sure 2 enumerated inputs is a no go, but can check that too.)

Seems like it is possible but won't run on the ANE (9/12/23).
"""

C = 768 # channels
T = 5_000 # vocab size, smaller than this + gather => ANE fails to build the network

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(T, C) # 100 tokens, 10 dimension embeddings
        self.emb.weight.data.uniform_(-1, 1)

        self.ls = nn.ModuleList([
            nn.Linear(C, C*2),
            nn.GELU(),
            nn.Linear(C*2, C*3),
            nn.GELU(),
            nn.Linear(C*3, C*2),
            nn.GELU(),
            nn.Linear(C*2, C),
        ])
        self.ln_out = nn.Linear(C, T, bias=False)
    def forward(self, x, y):
        """
        x: [1, n]
        y: [1, m, C]

        x is a proxy for a sequence of tokens
        y is a proxy for a KV cache
        """
        s = self.emb(x) + y
        # s = (s + y) / 2
        for l in self.ls:
            s = l(s)
        for l in self.ls:
            s = l(s)
        s = self.ln_out(s)
        # z = s
        # z = s * s / (s+s)
        # z = nn.functional.softmax(z)
        # z = z * s / (s+s)
        # z = nn.functional.layer_norm(z, z.shape[1:])
        # z = nn.functional.softmax(z)
        return s

    def foo(self, x):
        # x: [1, m] where m == 2n or m == n+1
        y = x.split(2)
        return y




net = Net()
net.eval()
trace_input = (torch.randint(0, 100, (1, 128)), torch.randn((1, 128, C)))
print(net(*trace_input))
traced = torch.jit.trace(net, trace_input)

print(f"traced: {traced}")

x_fixed_shape = ct.Shape(shape=[1, 256])
x_flex_shape = ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=1024, default=1)))
x_enum_shape = ct.EnumeratedShapes(shapes=[
    [1, 256],
    [1, 512],
    [1, 1024],
], default=[1,1])
y_enum_shape = ct.EnumeratedShapes(shapes=[
    [1, 256, C],
    [1, 512, C],
    [1, 1024, C],
], default=[1,256,C])
y_fixed_shape = ct.Shape(shape=[1, 128, C])

convert_args = {
    "inputs": [
        ct.TensorType("x", shape=x_enum_shape, dtype=np.int32),
        ct.TensorType("y", shape=y_enum_shape, dtype=np.float32),
    ],
    "outputs": [
        ct.TensorType("out", dtype=np.float32),
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

mlpackage_path = "enumerated-and-flexible-inputs.mlpackage"
model.save(mlpackage_path)
# reload ?
model = ct.models.model.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

all_default_input = {"x": torch.randint(0, 100, (1, 1)).int(), "y": torch.randn((1, 256, C)).float()}
all_enumerated_input = {"x": torch.randint(0, 100, (1, 256)).int(), "y": torch.randn((1, 256, C)).float()}
mixed_input = {"x": torch.randint(0, 100, (1, 512)).int(), "y": torch.randn((1, 512, C)).float()} # x enumerated, y flexible

print("PID:", os.getpid())
input("prese enter to go")
print()

signposter = Signposter("com.example.my_subsystem", Signposter.Category.PointsOfInterest)

# shapes = {k: v.shape for k,v, in all_default_input.items()}
print(f"predicting default")
end_interval = signposter.begin_interval("default shapes")
default_output = model.predict(all_default_input)["out"]
end_interval("end default shapes")
print(f"default shapes output: {default_output}")

print(f"predicting enum")
end_interval = signposter.begin_interval("enumerated shapes")
enumerated_output = model.predict(all_enumerated_input)["out"]
end_interval("end enumerated shapes")
print(f"enumerated shapes output: {enumerated_output}")

print(f"predicting mixed")
end_interval = signposter.begin_interval("mixed shapes")
mixed_output = model.predict(mixed_input)["out"]
end_interval("end mixed shapes")
print(f"mixed shapes output: {mixed_output}")
