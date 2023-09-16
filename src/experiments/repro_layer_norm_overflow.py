import torch
import torch.nn as nn
import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np


"""
Reproduce the Neural Engine layer_norm that overflows float16 when computed along
the first axis, but not the same norm computed along the third axis.

Guess what? This seems to be fixed on macOS Sonoma! You have to push B,C,S
higher (100,1024,1024) and tweak the scale factor (which is RNG dependent anyways)
to get it onto the ANE, but it doesn't overflow! Interestingly I think it might be
a little slower but I did not profile extensively.
"""

eps = 1e-5

# 100 works on both axes
# 10_000 overflows on axes=[1] only
# 1_000_000 overflows on axes=[1] or axes=[3]
scale_factor = 10_000

B,C,S = 1,4,2
g,b = 1, 0

# Use the built in MIL op.
@mb.program(input_specs=[mb.TensorSpec(shape=(B,C,1,S)),])
def built_in_axes_one_prog(x):
    gamma = (torch.ones((C,), dtype=torch.float32) * g).tolist()
    beta = (torch.ones((C), dtype=torch.float32) * b).tolist()
    y = mb.layer_norm(x=x, axes=[1], gamma=gamma, beta=beta, name="y")
    return y

# Use the built in MIL op, but transpose before and after.
@mb.program(input_specs=[mb.TensorSpec(shape=(B,C,1,S)),])
def built_in_axes_three_prog(x):
    gamma = (torch.ones((C,), dtype=torch.float32) * g).tolist()
    beta = (torch.ones((C), dtype=torch.float32) * b).tolist()
    x = mb.transpose(x=x, perm=[0,3,2,1]) # (B,S,1,C)
    x = mb.layer_norm(x=x, axes=[3], gamma=gamma, beta=beta)
    y = mb.transpose(x=x, perm=[0,3,2,1], name="y")
    return y

def make_model(prog, compute_units, precision=None):
    return ct.convert(
        prog,
        inputs=[ct.TensorType(name="x", shape=(B, C, 1, S), dtype=np.float16)],
        outputs=[ct.TensorType(name="y", dtype=np.float16)],
        compute_units=compute_units,
        compute_precision=precision,
        minimum_deployment_target= ct.target.iOS16,
        convert_to="mlprogram",
    )

def predict(model, x):
    return torch.from_numpy(model.predict({"x": x.numpy()})["y"])

unit = ct.ComputeUnit.CPU_AND_NE
precision = ct.precision.FLOAT16
axes_one_model = make_model(built_in_axes_one_prog, unit, precision)
axes_three_model = make_model(built_in_axes_three_prog, unit, precision)

# torch equivalent of built_in_axes_three_prog to use as a baseline
def torch_ln(x):
    """
    x (B,C,1,S)
    """
    x = x.permute(0, 3, 2, 1) # (B,S,1,C)
    y = nn.functional.layer_norm(x, [C], eps=1e-5)
    y = y.permute(0, 3, 2, 1) # (B,C,1,S)
    return y

rng = np.random.default_rng(seed=42)
x = torch.from_numpy(rng.normal(size=(B, C, 1, S))).float() * scale_factor

print("x", x)
print("x has any sums that overflow float16?", torch.any(x.sum(dim=1) > 65504).item())

axes_one_out = predict(axes_one_model, x)
print("axes=[1]\n-----\n",axes_one_out,"\n")

axes_three_out = predict(axes_three_model, x)
print("axes=[3]\n-----\n",axes_three_out,"\n")

baseline_out = torch_ln(x)
print("baseline\n-----\n",baseline_out,"\n")


print()
print("axes=[1] shape", axes_one_out.shape)
print("axes=[3] shape", axes_three_out.shape)
print("baseline_out shape", baseline_out.shape)

print()
print("baseline vs axes=[1]: close (1e-2)?", torch.allclose(baseline_out, axes_one_out, atol=1e-2, rtol=1e-4))
print("baseline vs axes=[1]: equal?", torch.all(baseline_out == axes_one_out))

print()
print("baseline vs axes=[3]: close (1e-2)?", torch.allclose(baseline_out, axes_three_out, atol=1e-2, rtol=1e-4))
print("baseline vs axes=[3]: equal?", torch.all(baseline_out == axes_three_out))