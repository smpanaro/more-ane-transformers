import torch
import torch.nn as nn
import os
import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np
from os_signpost import Signposter
from stopwatch import Stopwatch


"""
Test to see if there is special behavior for the MIL layer_norm
depending on which axes are passed.

Specifically:
- Is layer_norm more suspectible to overflow on certain axes?
- Is layer_norm on axes=[1] equivalent to LayerNormANE?

Answer: The axes=[1] version overflows before the axes=[3] version. Fixed in Sonoma though.
Question: Are they really doing the same thing? Seems so. Dropping the magnitude
of the input will make them equal (and they're not all zero).
"""

eps = 1e-5

# B,C,S = 1,4,3
B,C,S = 1,1024,512
g,b = 1, 0

# Use the built in MIL op.
@mb.program(input_specs=[mb.TensorSpec(shape=(B,C,1,S)),])
def built_in_axes_one_prog(x):
    gamma = (torch.ones((C,), dtype=torch.float32) * g).tolist()
    beta = (torch.ones((C), dtype=torch.float32) * b).tolist()
    y = mb.layer_norm(x=x, axes=[1], gamma=gamma, beta=beta, name="y")
    return y

# Use the built in MIL op, but transpose before and after. (You could write this in torch.)
@mb.program(input_specs=[mb.TensorSpec(shape=(B,C,1,S)),])
def built_in_axes_three_prog(x):
    gamma = (torch.ones((C,), dtype=torch.float32) * g).tolist()
    beta = (torch.ones((C), dtype=torch.float32) * b).tolist()
    x = mb.transpose(x=x, perm=[0,3,2,1]) # (B,S,1,C)
    x = mb.layer_norm(x=x, axes=[3], gamma=gamma, beta=beta)
    y = mb.transpose(x=x, perm=[0,3,2,1], name="y")
    return y

# Translate the custom LayerNormANE from ml-ane-transformers into MIL.
@mb.program(input_specs=[mb.TensorSpec(shape=(B,C,1,S)),])
def ane_prog(x):
    #channels_mean = inputs.mean(dim=1, keepdims=True)
    channels_mean = mb.reduce_mean(x=x, axes=[1], keep_dims=True)
    #zero_mean = inputs - channels_mean
    zero_mean = mb.sub(x=x, y=channels_mean)
    #zero_mean_sq = zero_mean * zero_mean
    zero_mean_sq = mb.mul(x=zero_mean, y=zero_mean)
    #denom = (zero_mean_sq.mean(dim=1, keepdims=True) + self.eps).rsqrt()
    zero_mean_sq_mean = mb.reduce_mean(x=zero_mean_sq, axes=[1], keep_dims=True)
    #zero_mean_sq_mean_plus_eps = mb.add(x=zero_mean_sq_mean, y=1e-5)
    denom = mb.rsqrt(x=zero_mean_sq_mean, epsilon=1e-5)
    #out = zero_mean * denom
    return mb.mul(x=zero_mean, y=denom, name="y")

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
ane_model = make_model(ane_prog, unit, precision)
# baseline_model = make_model(ane_prog, ct.ComputeUnit.CPU_ONLY, ct.precision.FLOAT32)

def torch_ln(x):
    """
    x (B,C,1,S)
    """
    x = x.permute(0, 3, 2, 1)
    y = nn.functional.layer_norm(x, [C], eps=1e-5)
    y = y.permute(0, 3, 2, 1)
    return y


# Avoid arange since it introduces patterns which seems not ideal
# x = torch.arange(B*C*S).reshape(B,C,1,S).float() / 0.1
# Use uniform to check for any odd behavior from the sequentialness of arange.
# x = torch.from_numpy(np.random.uniform(low=-10, high=10, size=(B, C, 1, S))).float()
rng = np.random.default_rng(seed=42)
x = torch.from_numpy(rng.normal(size=(B, C, 1, S))).float() * 10

# print("x",x)

signposter = Signposter("com.example.my_subsystem", Signposter.Category.PointsOfInterest)

print("PID:", os.getpid())
input("prese enter to go")


def profile_predict(model, x):
    sw = Stopwatch(3)
    sw.stop()
    sw.reset()

    for i in range(128):
        _ = predict(model, x) # warm up
    sw.start()
    for i in range(512):
        _ = predict(model, x)
    sw.stop()
    print(f"elapsed: {sw}")

end_interval = signposter.begin_interval("built in axes=[1]")
axes_one_out = predict(axes_one_model, x)
print("axes=[1]\n-----\n",axes_one_out,"\n")
profile_predict(axes_one_model, x)
end_interval("end")

end_interval = signposter.begin_interval("built in axes=[3]")
axes_three_out = predict(axes_three_model, x)
print("axes=[3]\n-----\n",axes_three_out,"\n")
profile_predict(axes_three_model, x)
end_interval("end")

end_interval = signposter.begin_interval("ane")
ane_out = predict(ane_model, x)
print("LayerNormANE\n-----\n",ane_out,"\n")
profile_predict(ane_model, x)
end_interval("end")

end_interval = signposter.begin_interval("baseline")
baseline_out = torch_ln(x)
print("baseline\n-----\n",baseline_out,"\n")
end_interval("end baseline")

print()
print("axes=[1] shape", axes_one_out.shape)
print("axes=[3] shape", axes_three_out.shape)
print("ane shape", ane_out.shape)

print()
print("axes=[3] vs axes=[1]: close (1e-5)?", torch.allclose(axes_three_out, axes_one_out, atol=1e-5, rtol=1e-4))
print("axes=[3] vs axes=[1]: equal?", torch.all(axes_three_out == axes_one_out))

print()
print("ane vs axes=[1]: close (1e-5)?", torch.allclose(ane_out, axes_one_out, atol=1e-5, rtol=1e-4))
print("ane vs axes=[1]: equal?", torch.all(ane_out == axes_one_out))

print()
print("baseline vs axes=[1]: close (1e-2)?", torch.allclose(baseline_out, axes_one_out, atol=1e-2, rtol=1e-4))
print("baseline vs axes=[1]: equal?", torch.all(baseline_out == axes_one_out))

print()
print("baseline vs axes=[3]: close (1e-2)?", torch.allclose(baseline_out, axes_three_out, atol=1e-2, rtol=1e-4))
print("baseline vs axes=[3]: equal?", torch.all(baseline_out == axes_three_out))