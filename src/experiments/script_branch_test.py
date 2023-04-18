import torch
from torch import nn
from torch.nn import functional as F
import coremltools as ct
from typing import Optional

from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb

# @register_torch_op
# def is_nonzero(context, node):
#     x = _get_inputs(context, node, expected=1)[0]
#     x = mb.elu(x=x, alpha=1.6732632423543772)
#     x = mb.mul(x=x, y=1.0507009873554805, name=node.name)
#     context.add(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(5, 800_000)
        # nn.init.zeros_(self.lin.weight)
        # nn.init.zeros_(self.lin.bias)

    def forward(self, x, cache: Optional[torch.Tensor]=None):
        # cache is either none or a tensor the same shape as x
        assert cache is None or x.shape == cache.shape

        # @torch.jit.script_if_tracing
        # def _add_cache(x, cache: torch.Tensor):
        #     # if cache is None:
        #     if torch.mean(cache) == 0:
        #         return x
        #     return cache + x + cache

        # _add_cache(x, cache)
        if cache is not None:
            x = cache + x

        return self.lin(x), torch.stack([x,x])

class CachedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cached = Model().eval()
        self.full = self.cached
        self.cached = torch.jit.trace(self.cached, (torch.randn(1,3,5), torch.randn(1,3,5)))
        self.full = torch.jit.trace(self.full, (torch.randn(1,3,5)))

    def forward(self, x, cache):
        if torch.min(cache) == 0:
            x, nc = self.full(x)
        else:
            x, nc = self.cached(x, cache)
        return x, nc

class CachedModelList(nn.ModuleList):
    def forward(self, x, cache):
        if torch.max(cache) != 0 and torch.min(cache) != 0:
            x = self[0](x, cache)
        else:
            x = self[1](x)
        return x


# cached = Model().eval()
# full = cached
# scripted_model = torch.jit.trace(cached, (torch.randn(1,3,5), torch.randn(1,3,5)))
# full = torch.jit.trace(full, (torch.randn(1,3,5)))
# cached.lin.weight = full.lin.weight
# print(scripted_model.code)

# model = CachedModelList([cached, full]).eval()
model = CachedModel().eval()

example_x = torch.randn(1,3,5)
example_cache = torch.zeros(1,3,5)
scripted_model = torch.jit.script(model, example_inputs=[(torch.randn(1,3,5), torch.zeros(1,3,5)), (torch.randn(1,3,5), torch.randn(1,3,5))])
print(scripted_model)
print(scripted_model.state_dict())


# model = Model().eval()
# scripted_model = torch.jit.trace(model, (torch.randn(1,3,5), torch.zeros(1,3,5)))

# print(scripted_model.code)
# print("with cache", scripted_model(torch.randn(1,3,5), torch.randn(1,3,5)))
# print("no cache", scripted_model(torch.randn(1,3,5), torch.zeros(1,3,5)))

# x = torch.randn(1,3,5)
# print(scripted_model(x))
# print(scripted_model(x, x))

cache_shape = ct.EnumeratedShapes(shapes=([1, 1,5], [1, 3, 5]), default=None)
import coremltools
mlmodel = coremltools.converters.convert(
  scripted_model,
  inputs=[
    ct.TensorType(name="x", shape=(1, 3, 5)),
    # ct.TensorType(name="use_cache", shape=([1]), dtype=ct.converters.mil.mil.types.bool),
    ct.TensorType(name="cache", shape=cache_shape),
  ],
  outputs=[
    ct.TensorType(name="out"),
    ct.TensorType(name="newcache")
  ],
    convert_to="milinternal",
)
# print(mlmodel._spec)
print(mlmodel)
# print(dir(mlmodel))
# main_block = mlmodel.functions["main"]

#approach 1
# def print_ops(op):
#     for output in op.outputs:
#         for child_op in output.child_ops:
#             print_ops(child_op)
#     if hasattr(op, "weight"):
#         print(op.weight, "|", op.weight.name, "|", op.op_type)
#         print(dir(op.weight), op.weight.sym_type, op.weight.val, op.weight.val.__class__)
#         op.weight.set_name(op.weight.name.replace("cached", "full"))
#         print("after", op.weight)
#     # else:
    #     print(op.name, dir(op))

# for op in main_block.operations:
#     print_ops(op)

# approach 2

# ops = main_block.find_ops(op_type="linear")
# full_ops = [o for o in ops if hasattr(o, "weight") and o.weight.name.startswith("full")]
# full_ops = {o.weight.name:o for o in full_ops}

# cached_ops = [o for o in ops if hasattr(o, "weight") and o.weight.name.startswith("cached")]

# boundary_op = main_block.operations[0]
# print("boundary", boundary_op.name)

# with main_block:
#     for op in cached_ops:
#         print('ayy', op.name)
#         if hasattr(op, "weight"):
#             print(op.weight)
#             full_op = full_ops[op.weight.name.replace("cached", "full")]
#             if not full_op:
#                 continue
#             main_block.replace_uses_of_var_after_op(
#                 anchor_op=boundary_op,
#                 old_var=op.weight,
#                 new_var=full_op.weight,
#                 # force_replace=True # For quantized models.
#             )


# print(mlmodel)

mlp = ct.convert(mlmodel, convert_to="mlprogram")

pred = mlp.predict({"x": example_x.float().numpy(), "cache": example_cache.float().numpy()})
print("predicted", pred)

mlp.save("script-branch-model.mlpackage")

# class _LoopBody(nn.Module):
#     def __init__(self, channels):
#         super(_LoopBody, self).__init__()
#         conv = nn.Conv2d(
#             in_channels=channels,
#             out_channels=channels,
#             kernel_size=3,
#             padding=1,
#         )
#         self.conv = conv

#     def forward(self, x):
#         x = self.conv(x)
#         x = F.relu(x)
#         return x

# class ControlFlowNet2(nn.Module):
#     def __init__(self, num_channels: int):
#         super(ControlFlowNet2, self).__init__()
#         self.loop_body = _LoopBody(num_channels)
#         self.loop_body = torch.jit.trace(self.loop_body, torch.randn(1,3,64,64))

#     def forward(self, x):
#         avg = torch.mean(x)
#         if avg.item() < 0:
#             loop_count = 2
#         else:
#             loop_count = 1
#         for _ in range(loop_count):
#             x = self.loop_body(x)
#         return x

# model = ControlFlowNet2(num_channels=3)
# scripted_model = torch.jit.script(model)


# mlmodel = ct.converters.convert(
#   scripted_model,
#   inputs=[ct.TensorType(shape=(1, 3, 64, 64))],
# )

# mlmodel.save("trace-script-example.mlpackage")