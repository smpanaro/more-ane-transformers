import torch
from torch import nn
import numpy as np
from src.ml_ane_transformers.ane.layer_norm import LayerNormANE as LayerNorm
from src.ml_ane_transformers.ane.kahan_layer_norm import KahanLayerNormANE as KahanLayerNorm
import coremltools as ct
from src.utils.psnr import compute_psnr
from coremltools.converters.mil import Builder as mb
import sys

"""
Compare and test a Kahan summation implementation of layer norm vs.
the default ANE-optimized one.

Seems that for some reason my Kahan implementation is slightly less accurate.

Ended up finding that the model builder layer_norm can take a 4D input and do
the norm on the channel dimension. Unfortunately doesn't work on the Neural Engine afaict.
"""

torch.manual_seed(42)

B,C,S = 1, 1024, 512
# B,C,S = 1, 6, 1
# B,C,S = 1,3,1
# B,C,S = 1,3,2
# x = torch.FloatTensor(B,C,1,S).uniform_(torch.finfo(torch.half).min*0.9, torch.finfo(torch.half).max*0.9)
x = torch.randn((B,C,1,S), dtype=torch.float16).float().cpu()
# x = torch.tensor([[[[10000.0]], [[3.14159]], [[2.71828]]]], dtype=torch.float16).float().cpu()
# x = torch.tensor([[[[10000.0, 2.71828]], [[3.14159, 10000.0]], [[2.71828, 3.14159]]]], dtype=torch.float16).float().cpu()
# print(x.shape, x.to("mps").half().cumsum(dim=1))

# Ignore learnable params.
clip_mag = None#1e7
ln = LayerNorm(C, clip_mag=clip_mag, elementwise_affine=False)
kln = KahanLayerNorm(C, clip_mag=clip_mag, elementwise_affine=False)
nnln = nn.LayerNorm(C, elementwise_affine=False)

def print_stats(normal, kahan):
    assert normal.shape == kahan.shape
    print("all close?", torch.allclose(normal, kahan))
    print("equal?", torch.equal(normal, kahan))
    print("mean diff", torch.mean(normal - kahan))
    print("max diff", torch.max(torch.abs(normal - kahan)))
    # print("psnr", compute_psnr(normal, kahan))
    print("psnr", compute_psnr(kahan, normal))
    print("num close:", torch.sum(torch.isclose(normal, kahan)))

with torch.no_grad():
    km = kln.kahan_mean(x.to("mps").half(), 4).float().cpu()
    hm = x.to("mps").half().mean(dim=1, keepdim=True).float().cpu()
    m = x.to("mps").float().mean(dim=1, keepdim=True).float().cpu()
    dm = x.double().mean(dim=1, keepdim=True)

print("mean vs kahan mean half\n----")
print_stats(m, km)
print_stats(m, hm)
# print("kahan", km)
# print("exactly:", m)

with torch.no_grad():
    ln_res = ln(x.float())
    kln_res = kln(x.float())
# print("float32\n----")
# print_stats(ln_res, kln_res)

with torch.no_grad():
    y = x.half().to("mps")
    ln_res_half = ln(y).float().cpu()
    kln_res_half = kln(y).float().cpu()
# print("\nfloat16\n----")
# print_stats(ln_res_half, kln_res_half)

print("\nfloat16 normal v float32 normal\n----")
print_stats(ln_res, ln_res_half)

print("\nfloat16 kahan v float32 normal\n----")
print_stats(ln_res, kln_res_half)

def convert_bc1s_norm(n, f32=False, skip_trace=False):
    if not skip_trace:
        traced = torch.jit.trace(n, (x,))
        mlp = ct.convert(traced,
                        inputs=[ct.TensorType(name="x", shape=(B,C,1,S), dtype=np.float32)],
                        outputs=[ct.TensorType(name="y", dtype=np.float32)],
                        compute_precision=ct.precision.FLOAT32 if f32 else ct.precision.FLOAT16,
                        compute_units=ct.ComputeUnit.CPU_AND_NE,
                        convert_to="milinternal")
    else:
        mlp = n
    print(n.__class__)
    print(mlp)
    return ct.convert(mlp,
                    compute_precision=ct.precision.FLOAT32 if f32 else ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_AND_NE,
                    convert_to="mlprogram")

def convert_bsc_norm(n, f32=False):
    traced = torch.jit.trace(n, (x.permute(0,3,1,2).squeeze(-1),))
    mlp = ct.convert(traced,
                    inputs=[ct.TensorType(name="x", shape=(B,S,C), dtype=np.float32)],
                    outputs=[ct.TensorType(name="y", dtype=np.float32)],
                    compute_precision=ct.precision.FLOAT32 if f32 else ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_AND_NE,
                    convert_to="milinternal")
    print(n.__class__)
    print(mlp)
    return ct.convert(mlp,
                    compute_precision=ct.precision.FLOAT32 if f32 else ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_AND_NE,
                    convert_to="mlprogram")

# Interesting...
@mb.program(input_specs=[mb.TensorSpec(shape=(B,C,1,S)),])
def ln_prog(x):
    # x = mb.squeeze(x=x, axes=[2], name='squeeze')
    x = mb.layer_norm(x=x, axes=[1], name="y")
    # x = mb.expand_dims(x=x, axes=[2], name="y")
    return x


cln = convert_bc1s_norm(ln, False)
# ckln = convert_bc1s_norm(kln, False)
lnp = convert_bc1s_norm(ln_prog, False, skip_trace=True)
# half_nnln = convert_bsc_norm(nn.LayerNorm(C, elementwise_affine=False))
# nnln = convert_bsc_norm(nn.LayerNorm(C, elementwise_affine=False),f32=True)

inp = {"x": x.float().numpy()}
coreml_ln = torch.from_numpy(cln.predict(inp)["y"])
coreml_kln = torch.from_numpy(ckln.predict(inp)["y"])
print(lnp.predict(inp))
coreml_lnp = torch.from_numpy(lnp.predict(inp)["y"])
coreml_half_nnln = half_nnln.predict({"x": x.permute(0,3,1,2).squeeze(-1).float().numpy()})["y"]
coreml_half_nnln = torch.from_numpy(coreml_half_nnln).permute(0,2,1).unsqueeze(2)
coreml_nnln = nnln.predict({"x": x.permute(0,3,1,2).squeeze(-1).float().numpy()})["y"]
coreml_nnln = torch.from_numpy(coreml_nnln).permute(0,2,1).unsqueeze(2)

print("\coreml nn ln vs kln\n----")
print_stats(coreml_nnln, coreml_kln)
print("\coreml nn ln vs ln\n----")
print_stats(coreml_nnln, coreml_ln)
print("\coreml nn ln vs half nn\n----")
print_stats(coreml_nnln, coreml_half_nnln)
print("\coreml nn ln vs ln prog\n----")
print_stats(coreml_nnln, coreml_lnp)


# Output of coreml norms for 1x1024x1x512 input with a 512 chunks.
# Took forever to run and I think basically shows that Kahan accumulates too much error.
# \coreml nn ln vs kln
# ----
# all close? False
# equal? False
# mean diff tensor(2.1594e-06)
# max diff tensor(0.0187)
# psnr 67.48296284743999
# num close: tensor(2398)
# \coreml nn ln vs ln
# ----
# all close? False
# equal? False
# mean diff tensor(2.3021e-06)
# max diff tensor(0.0057)
# psnr 77.59771092144952
# num close: tensor(7922)

