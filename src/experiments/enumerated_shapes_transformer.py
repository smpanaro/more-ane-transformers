import coremltools as ct
import torch
from torch import nn
import numpy as np
from dataclasses import dataclass

"""
See if there's a way to achieve KV-caching in one model by using
enumerated shapes cleverly.

See multi_variable_inputs experiment for a cleaner version.
"""

from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb

del _TORCH_OPS_REGISTRY["constantchunk"]

# S K E T C H Y
# Basically avoids the safety checks that are in place and we
# make sure not to pass anything in that doesn't divide evenly.
@register_torch_op
def constantchunk(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    # ConstantChunk gets its parameters as attributes of the node.
    chunks = node.attr["chunks"]
    dim = node.attr["dim"]

    res = mb.split(x=x, num_splits=chunks, axis=dim, name=node.name)
    for val, name in zip(res, node.outputs):
        context.add(val, name)

# @register_torch_op
# def chunk(context, node):
#     inputs = _get_inputs(context, node, expected=3)
#     x = inputs[0]
#     # ConstantChunk gets its parameters as attributes of the node.
#     chunks = inputs[1]
#     dim = inputs[2]

#     res = mb.split(x=x, num_splits=chunks, axis=dim, name=node.name)
#     for val, name in zip(res, node.outputs):
#         context.add(val, name)


# Q1: Is it possible to crop everything to the right size in-model?
# Q2: Can we apply KV-caching in the attention block always / conditionally / at all?

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_up = nn.Linear(5, 3000)
        self.linear_down = nn.Linear(3000, 5)

    def forward1(self, ids, output_mask, kv_cache, kv_mask):
        """
        n: the enumerated size
        *batch size fixed to 1 for simplicity
        ids: [1, n]
        output_mask: [1]
        kv_cache: [num_layers, 1, max(n)*2, hidden_size]
        kv_mask: [1, max(n), hidden_size]
        """

        # CoreML doesn't like passing inputs directly to outputs.
        ids = ids * 1
        output_mask = output_mask * 1
        kv_cache = kv_cache * 1
        kv_mask = kv_mask * 1

        k_cache, v_cache = kv_cache.chunk(2, dim=2) # each [num_layers, 1, max(n), hidden_size]
        k_cache = k_cache[:, :, [output_mask], :]
        v_cache = v_cache[:, :, [output_mask], :]
        kv_cache = torch.cat([k_cache, v_cache], dim=2)

        kv_mask = kv_mask[:, [output_mask], :]

        return ids, output_mask, kv_cache, kv_mask

    def forward(self, combo_input, input_mask):
        """
        n: the enumerated size

        id_embeds: [1, n, hidden_size]
        output_mask: [1]
        kv_cache: [num_layers, 1, n*2, hidden_size]
        kv_mask: [1, n, hidden_size]

        idea 1:
        enum_inputs[1] [num_layers+1, 1, n*2, hidden_size]
            idx[-1][0] = cat([id_embeds, kv_mask])
        output_mask: [1]

        idea 2:
        combo_input: [1, X, hidden_size]
        X = (num_layers*2*n) + n + (1 OR max n)
        """

        # linear, layer norm, gelu all operate row wise so it's ok to pull out the output_mask'th row
        # perform the op then put it back into a zeros matrix
        # will be basically impossible to debug of course
        # (note: this is a no go because conditionals push you off the neural engine -- unless
        # the inputs don't get pushed to CPU, but that would be hard to test)

        # I don't think this will work. id_embeds is (n, hidden_size) but most of the time
        # we only care about 1 row and we can't do a conditional since that makes shapes dynamic.


        # Wait.. we almost always want id_embeds to be (1, hidden_state). What if we fix that and
        # prepopulate the kv_cache? Guess that requires running the model once beforehand?

        # one input... [1, n * X, hidden_size]
        # first num_layers*2*n are the kv_cache
        # next n are the kv_mask
        # remainder are the inputs -- need to see if split or chunk support remainders..

        # num layers = 3
        # hidden size = 5
        splits = combo_input[:, :-input_mask, :].chunk(3+3+1, dim=1)
        pairs = [(splits[i], splits[i+1]) for i in range(0, 6, 2)]
        kv_caches = pairs[:3]
        kv_mask = splits[-1]
        input_embeds = combo_input[:, -input_mask:, :]

        kv_caches = torch.stack([torch.stack(kv) for kv in kv_caches]) # output transform

        B = torch.ones((3000,3000))
        C = torch.ones((3000, 3000))
        for _ in range(10):
            x = C @ B
        input_embeds = x
        # input_embeds = combo_input.reshape(-1).chunk(3+3+2)[0]
        # kv_caches = combo_input * 1
        # kv_mask = combo_input * 1

        return input_embeds, kv_caches, kv_mask

@dataclass
class Inputs:
    ids: torch.IntTensor
    output_mask: torch.IntTensor
    kv_cache: torch.FloatTensor
    kv_mask: torch.IntTensor

    def apply(self, net):
        return net(self.ids, self.output_mask, self.kv_cache, self.kv_mask)

    def shapes(self):
        return f"{[i.shape for i in [self.ids, self.output_mask, self.kv_cache, self.kv_mask]]}"

def build_inputs(n: int, output_idx: int):
    assert output_idx <= n
    ids = torch.arange(0, n, dtype=torch.int32)
    output_mask = torch.tensor([output_idx], dtype=torch.int32)
    max_n = 512
    kv_cache = torch.zeros((12, 1, max_n*2, 768))
    kv_mask = torch.ones((1, max_n*2, 768))
    return Inputs(
        ids=ids,
        output_mask=output_mask,
        kv_cache=kv_cache,
        kv_mask=kv_mask
    )

combo_inputs = {}
for n in [2, 4, 8]:
    combo_inputs[n] = torch.arange(0, 5*((2*3*n)+n+1)).float().view(1, -1, 5)
combo_inputs[512] = torch.arange(0, 5*((2*3*512)+512+512)).float().view(1, -1, 5)
def get_input_mask(n):
    if n == 512:
        return torch.tensor([512], dtype=torch.int32)
    return torch.tensor([1], dtype=torch.int32)
print("combo_inputs:", {k: v.shape for k,v in combo_inputs.items()})

net = Net().eval()

# inputs_one = build_inputs(1, 0)
# inputs_four = build_inputs(4, 2)

# for inp in [inputs_one, inputs_four]:
#     out = inp.apply(net)
#     print(f"inputs idx {inp.output_mask.item()}: {inp.shapes()}")
#     print(f"  -> {[o.shape for o in out]}")

for n, inp in combo_inputs.items():
    out = net(inp, get_input_mask(n))
    print(f"n: {n}")
    print(f" -> {[o.shape for o in out]}")

# inputs_six = build_inputs(6, 3)
# traced = torch.jit.trace(net, (inputs_six.ids, inputs_six.output_mask, inputs_six.kv_cache, inputs_six.kv_mask))
traced = torch.jit.trace(net, (combo_inputs[4], get_input_mask(4)))
all_shapes = [v.shape for v in combo_inputs.values()]
print("all shapes", all_shapes)
# print(traced.code)
mlprog = ct.convert(
    traced,
    inputs=[
        # ct.TensorType("ids", shape=ct.EnumeratedShapes([(1,1), (1,2), (1,3), (1,6)]), dtype=np.int32),
        # ct.TensorType("output_mask", shape=[1], dtype=np.int32),
        # ct.TensorType("kv_cache", shape=[12, 1, 1024, 768], dtype=np.float16),
        # ct.TensorType("kv_mask", shape=[1, 1024, 768], dtype=np.float16),
        ct.TensorType("combo_input", shape=ct.EnumeratedShapes(all_shapes, default=all_shapes[-1]), dtype=np.float16),
        ct.TensorType("input_mask", shape=[1], dtype=np.int32),
    ],
    outputs=[
        ct.TensorType("ids_out", dtype=np.int32),
        # ct.TensorType("output_mask_out", dtype=np.int32),
        ct.TensorType("kv_cache_out", dtype=np.float16),
        ct.TensorType("kv_mask_out", dtype=np.float16),
    ],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS16,
    convert_to="milinternal",
)
# print(mlprog)

mlmodel = ct.convert(mlprog,
                     compute_precision=ct.precision.FLOAT16,
                     minimum_deployment_target=ct.target.iOS16,
                     compute_units=ct.ComputeUnit.CPU_AND_NE,
                     convert_to="mlprogram")
mlmodel.save("enumerated-shapes-test.mlpackage")

input("attach debugger now")

for _ in range(1000):
    for n, inp in combo_inputs.items():
        out = mlmodel.predict({"combo_input": inp.numpy(), "input_mask": get_input_mask(n).numpy()})
        out_shapes = {k: v.shape for k,v in out.items()}
        # print(f"mlmodel n {n} -> {out_shapes}")
        # print({k:v for k,v in out.items()})