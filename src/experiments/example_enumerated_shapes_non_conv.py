import torch
from torch import Tensor, nn
import coremltools as ct
from timeit import default_timer as timer
from  os_signpost import Signposter
signposter = Signposter("com.smpanaro.more-ane-transformers", Signposter.Category.PointsOfInterest)

"""
See if I can reproduce https://github.com/apple/coremltools/issues/1909 on my machine.
If so, maybe this is why I can't get gpt2 working with enumerated shapes (!).

Result: Yes, layer_norm doesn't work with enumerated shapes unless you do it on a different axis.
Also, matrix multiplication doesn't work (but it does on iOS17).
"""

# If you want to try with different axis values, or by changing the inputs to be 4D, uncomment:
# from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op, _TORCH_OPS_REGISTRY
# from coremltools.converters.mil.frontend.torch.ops import _get_inputs
# from coremltools.converters.mil import Builder as mb
# from coremltools.converters.mil.mil.types.symbolic import any_symbolic
# del _TORCH_OPS_REGISTRY["layer_norm"]
# @register_torch_op
# def layer_norm(context, node):
#     inputs = _get_inputs(context, node, expected=6)
#     _input = inputs[0]
#     normalized_shape = inputs[1]
#     weight = inputs[2]
#     bias = inputs[3]
#     eps = inputs[4]
#     # cudnn_enable = inputs[5] unused

#     layer_norm = mb.layer_norm(
#         x=_input,
#         # axes=list(range(-len(normalized_shape.val), 0)),
#         axes=[-1],
#         gamma=weight,
#         beta=bias,
#         epsilon=eps,
#         name=node.name,
#     )
#     context.add(layer_norm)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.LayerNorm(2048)
        self.linear1    = nn.Linear(2048, 4*2048)
        self.linear2    = nn.Linear(4*2048, 2048)
        self.linear3 = nn.Linear(2048, 2*2048)

    def forward(
        self,
        x: Tensor,
    ):
        # For layer_norm.
        # x = self.ln(x) # if we comment out layerNorm, x_enum_shape model takes 15ms (use ane)
        # x = x @ torch.ones(2048, 2048)

        # Testing attention block matmul
        q,k = self.linear3(x).split(2048, dim=-1)
        B,T,C = k.size()
        n_head = 16

        # Normal matmul
        # k = k.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
        # q = q.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
        # x = q @ k.transpose(-1, -2) # (B, nh, T, T) # doesn't work
        # x = torch.einsum('bnth,bnsh->bnts', q, k) # doesn't work

        # ml-ane-transformers matmul
        q = q.transpose(1,2).view(B,C,1,T)
        k = k.transpose(1,2).view(B,C,1,T)
        mh_q = q.split(
            C // n_head,
            dim=1)
        mh_k = k.transpose(1, 3).split(
            C // n_head,
            dim=3)
        # attn_weights = [
        #     torch.einsum('bchq,bkhc->bkhq', [qi, ki]) #* self.q_normalize_fact
        #     for qi, ki in zip(mh_q, mh_k)
        # ]
        # x = torch.cat(attn_weights)
        x = torch.einsum('bchq,bkhc->bkhq', [mh_q[0], mh_k[0]])

        return x

module = MyModel()
module.eval()

n = 2048
input1 = torch.ones(1, 1, n)
input2 = torch.ones(1, 128, n)

traced_module = torch.jit.trace(module, (input1))

# choose "CPU and Neural Engine" in XCode Performance Report
# x_enum_shape: 175ms (won't use ane)
# x_range_shape: 20ms (use ane)
# x_fixed_shape: 20ms (use ane)
x_enum_shape = ct.EnumeratedShapes([input1.shape, input2.shape], default=input2.shape)
x_range_shape = ct.Shape((1, ct.RangeDim(1,128), n), default=(1, 1, n))
x_fixed_shape = input1.shape

x_input_shapes = [x_enum_shape, x_range_shape, x_fixed_shape]

for i, x_input_shape in enumerate(x_input_shapes):
    print(f"=== x_input_shape={x_input_shape} ===")
    mlmodel = ct.convert(
        traced_module,
        convert_to="mlprogram",
        # convert_to="milinternal",
        inputs=[ct.TensorType("x", x_input_shape)],
        outputs=[ct.TensorType(name="out_x")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    # print(mlmodel)
    # lkjl
    mlmodel.save(f"model_{i}.mlpackage")

    for i in range(3):
        startT = timer()
        prediciton = mlmodel.predict({'x': input2})
        print(f"predict time {timer() - startT: .3f}s")
    print("-")