import coremltools as ct
import torch
from torch import nn
import numpy as np

"""
I thought script_if_tracing did not work, but it seems to.
"""

@torch.jit.script_if_tracing
def double_if_negative(x):
    if x.sum() > 0:
        return x
    return x + x

class Net(nn.Module):
    def forward(self, x):
        x = x + 0.5
        x = double_if_negative(x)
        return x

net = Net().eval()
traced = torch.jit.trace(net, (torch.tensor([2]),))
# traced = torch.jit.script(net) # Also does not work.
print(traced.code)

milinternal = ct.convert(traced,
    inputs=[ct.TensorType("x", shape=[1], dtype=np.float32)],
    outputs=[ct.TensorType("y", dtype=np.float32)],
    minimum_deployment_target=ct.target.iOS16, # TODO: Is this needed?
    convert_to="milinternal"
)
print(milinternal)

mlmodel = ct.convert(milinternal, convert_to="mlprogram", compute_units=ct.ComputeUnit.CPU_ONLY)
pos_input = torch.tensor([3.])
neg_input = torch.tensor([-4.])
print("positive case:")
print(f"expected (traced torch): {pos_input} -> {traced(pos_input)}")
print(f"actual (mlmodel): {pos_input} -> {mlmodel.predict({'x': pos_input.numpy()})['y']}")

print("\nnegative case:")
print(f"expected (traced torch): {neg_input} -> {traced(neg_input)}")
print(f"actual (mlmodel): {neg_input} -> {mlmodel.predict({'x': neg_input.numpy()})['y']}")




