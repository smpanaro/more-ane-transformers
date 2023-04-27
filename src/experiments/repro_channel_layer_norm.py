import torch
import torch.nn as nn
import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np

"""
Reproduce layer norm on the first dimension of a 4D tensor that
works on CPU but fails on Neural Engine.
"""

eps = 1e-5

# B,C,S = 1,4,2 # This will look like it works because it's too small for the ANE.
B,C,S = 1,1800,16 # This will fail since it's large enough for the ANE. Make it bigger if it doesn't.

g,b = 1, 0 # It's not these, the results change predictably with them.
@mb.program(input_specs=[mb.TensorSpec(shape=(B,C,1,S)),])
def ln_prog(x):
    gamma = (torch.ones((C,), dtype=torch.float32) * g).tolist()
    beta = (torch.ones((C), dtype=torch.float32) * b).tolist()
    x = mb.layer_norm(x=x, axes=[1], gamma=gamma, beta=beta, name="y")
    return x

x = torch.arange(B*C*S).reshape(B,C,1,S).float()
# x = torch.cat([torch.ones((1,1,1,S)).float()] + [torch.zeros((1,1,1,S)).float() for i in range(C-1)], dim=1)
# x = torch.cat([torch.ones((1,C,1,1)).float() * C*2] + [torch.zeros((1,C,1,1)).float() for i in range(S-1)], dim=3)
# print(x.squeeze(2).permute(0,2,1))
# x = x[torch.randperm(x.shape[0])]
# x = torch.randn((B,C,1,S)).float() + 1000

def make_model(prog, compute_units):
    return ct.convert(
        prog,
        inputs=[ct.TensorType(name="x", shape=(B,C,1,S), dtype=np.float32)],
        outputs=[ct.TensorType(name="y", dtype=np.float32)],
        compute_units=compute_units,
        # iOS 16 doesn't help.
        convert_to="mlprogram",
    )

def predict(model, x):
    return torch.from_numpy(model.predict({"x": x.numpy()})["y"])

cpu_model = make_model(ln_prog, ct.ComputeUnit.CPU_ONLY)
ane_model = make_model(ln_prog, ct.ComputeUnit.CPU_AND_NE)

print("input\n-----\n",x,"\n")

def ane_layer_norm(x):
    channels_mean = x.mean(dim=1, keepdims=True)
    zero_mean = x - channels_mean
    zero_mean_sq = zero_mean * zero_mean
    denom = (zero_mean_sq.mean(dim=1, keepdims=True) + eps).rsqrt()
    return zero_mean * denom
ane_layer_norm_out = ane_layer_norm(x)
print("ml-ane-transformers LayerNorm result\n-----\n",ane_layer_norm_out,"\n")

cpu_out = predict(cpu_model, x)
print("mb.program CPU_ONLY result\n-----\n",cpu_out,"\n")

# Will be slightly different because of float16.
ane_out = predict(ane_model, x)
print("mb.program CPU_AND_NE result\n-----\n",ane_out,"\n")


def custom_norm(x, dim):
    channels_mean = x.mean(dim=dim, keepdims=True)  # B11S
    zero_mean = x - channels_mean                   # BC1S
    zero_mean_sq = zero_mean * zero_mean            # BC1S
    denom = (zero_mean_sq.mean(dim=dim, keepdims=True) + eps).rsqrt() # B11S
    return zero_mean * denom

# for i in range(len(x.shape)):
#     print(i, "result\n-----\n",custom_norm(x, i),"\n")
#     break

# x_norms = torch.norm(x, p=2, dim=1, keepdim=True)
# print(x_norms.shape)
# x_norm = x / ((x_norms + eps) * 2 * C)
# print(x_norms)

# Trying to find the pattern in how the ANE output is different.
print(cpu_out.mean())
print(B,C,S, (ane_out / cpu_out).mean())
# 1 1 2048 tensor(inf) (ane_out = inf, cpu_out = 0)
# 1 2 2048 tensor(3.9978)
# 1 3 2048 tensor(nan) (eyeballing it looks like ~8 but there's a bunch of zeros)
# 1 4 2048 tensor(8.9392)
# 1 5 2048 tensor(nan) (eyeballing it looks like ~1.26 but there's a bunch of zeros)
# 1 6 2048 tensor(13.6535)
# 1 7 2048 tensor(17.2813) # median
# 1 8 2048 tensor(18.3189)
# 1 10 2048 tensor(22.9630)
# 1 14 2048 tensor(32.2279)
# 1 15 2048 tensor(7249.1562) (maybe median is a better measure, that's 35)
# 1 16 2048 tensor(36.8549)
# 1 20 2048 tensor(46.0989)
# 1 30 2048 tensor(69.1986)

# 1 20 1024 tensor(23.0494)

# 1 1800 1 tensor(2.2537)
# 1 1800 2 tensor(4.2657)
# 1 1800 3 tensor(6.2848)
# 1 1800 4 tensor(8.3176)
# 1 1800 5 tensor(10.3583)
# 1 1800 16 tensor(32.6669)

# 1 3600 8 tensor(32.6650) # BCS (no 1)

# 1 3600 1 tensor(4.2657)