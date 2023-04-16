import torch
from torch import nn
import coremltools as ct
import numpy as np

"""
Trying to find a way to replace a row in a matrix that can run on the Neural Engine.
"""

# new_q, new_k, new_v = q,k,v # new_q, new_k, new_v = [B, 1, n_embd]
# old_k, old_v = kv_cache.chunk(2, dim=1) # each (B, T, C)

class Net(nn.Module):
    def forward(self, new_k, old_k, mask):
        # new_k [B, 1, n_embd]
        # old_k [B, T, n_mbd]
        # insert_at [1]

        # mask = torch.zeros_like(old_k)
        # mask[:, insert_at, :] = 1 # Using a symbol here fails an assert, the first : is also not cool.


        # ValueError: Type mismatch <class 'coremltools.converters.mil.mil.types.type_tensor.tensor.<locals>.tensor'> vs. <class 'coremltools.converters.mil.mil.types.type_tensor.tensor.<locals>.tensor'>
        # mask = torch.zeros_like(old_k)
        # new_mask = torch.ones_like(new_k)
        # mask = torch.cat([mask[:, :insert_at, :], new_mask, mask[:, insert_at+1:, :]], dim=1)

        # mask = torch.zeros_like(old_k)
        # new_mask = torch.ones_like(new_k)
        # mask = torch.cat([mask[:, :2, :], new_mask, mask[:, 2+1:, :]], dim=1)

        return torch.where(mask.bool(), new_k, old_k)

net = Net().eval()

new_k = torch.tensor([[[99.,98,97]]])
old_k = torch.tensor([[[0.,1,2],[3,4,5],[6,7,8],[9,10,11]]])
insert_at = torch.tensor([2])

mask = torch.zeros_like(old_k)
mask[:, insert_at, :] = 1
print(mask.shape)

result = net(new_k, old_k, mask)
print(result)
# assert torch.equal(result[0, 2, :], new_k.squeeze())

traced = torch.jit.trace(net, (new_k, old_k, mask))

prog = ct.convert(traced,
    inputs=[
        ct.TensorType(name="new_k", shape=new_k.shape, dtype=np.float16),
        ct.TensorType(name="old_k", shape=old_k.shape, dtype=np.float16),
        # ct.TensorType(name="insert_at", shape=insert_at.shape, dtype=np.int32),
        ct.TensorType(name="mask", shape=insert_at.shape, dtype=np.float16),
    ],
    outputs=[
        ct.TensorType(name="result", dtype=np.float16),
    ],
    minimum_deployment_target=ct.target.iOS16, # float16 for less casting
    convert_to="milinternal")
print(prog)

mlmodel = ct.convert(prog,
    minimum_deployment_target=ct.target.iOS16, # float16 for less casting
    convert_to="mlprogram")

mlmodel.save("replace-row-model.mlpackage")