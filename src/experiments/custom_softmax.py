import torch
import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np
from coremltools.converters.mil.mil import types

"""
Confirm that coremltools uses the softmax subtraction trick (it must, right?).
"""

S = 3
@mb.program(input_specs=[mb.TensorSpec(shape=(S,), dtype=types.fp16),], opset_version=ct.target.iOS16)
def sm_prog(x):
    return mb.softmax(x=x, axis=-1, name="y")

f32 = False
mlmodel = ct.convert(sm_prog,
                    inputs=[ct.TensorType(name="x", shape=(S,), dtype=types.fp16)],
                    outputs=[ct.TensorType(name="y", dtype=types.fp16)],
                    compute_precision=ct.precision.FLOAT32 if f32 else ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_ONLY,
                    minimum_deployment_target=ct.target.iOS16,
                    convert_to="mlprogram")
# print(mlmodel)

x = torch.tensor([700_000, 700_000, 30_000]).float()
inputs = {"x": x.numpy()}
ml_y = mlmodel.predict(inputs)["y"]

torch_y = x.softmax(-1)

manual_y = torch.exp(x - x.max()) / torch.exp(x-x.max()).sum(-1, keepdim=True)

print("ml_y", ml_y)
print("torch_y", torch_y)
print("manual_y", manual_y)
