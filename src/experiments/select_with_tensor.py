import coremltools as ct
import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from os_signpost import Signposter
import os

"""
Test to see if you can select from a tensor using indices computed from
a 0D tensor.

Result: Yes, but it only runs on CPU.
"""

class Net(nn.Module):
    def forward(self, x, index):
        """
        x: [layer, batch, seq, dim]
        index: [1]
        """
        newx = []
        for idx, l in enumerate(x):
            newx.append((l + idx))

        fullx = torch.stack(newx)
        return fullx[:,:,index:-2, :]

# x_shape = [12,1,512,768*2]
x_shape = [2,1,5,3]
prod = np.prod(x_shape)
x = (torch.arange(prod).reshape(x_shape) / prod).half()
index = torch.tensor([1]).int()
net = Net()
net.eval()

print(x)
print("torch", net(x, index))


#trace
traced_model = torch.jit.trace(net, (x, index))
mlprog = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="x", shape=x.shape, dtype=np.float16),
        ct.TensorType(name="index", shape=[1], dtype=np.int32),
    ],
    outputs=[
        ct.TensorType(name="y", dtype=np.float16),
    ],
    convert_to="milinternal",
    minimum_deployment_target=ct.target.iOS16,
)

print(mlprog)

model = ct.convert(
    mlprog,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS16,
    # pass_pipeline=ct.PassPipeline.EMPTY,
    # **convert_args,
)

out = model.predict({"x": x, "index": index.int()})
print("coreml", out)

# this
# %3_item: (int32)(Scalar) = squeeze(x=%index, name="3_item")
# %concat_0: (4,int32)^(Tensor) = concat(values=(%concat_0_values0_0, %concat_0_values1_0, %3_item, %concat_0_values3_0), axis=0, interleave=False, name="concat_0")
# %y: (is30, 1, is31, is32, fp16)(Tensor) = slice_by_index(x=%x, begin=%concat_0, end=[2, 1, -2, 3], end_mask=[True, True, False, True], name="17_cast")

# gpt2
# %51_item: (int32)(Scalar) = squeeze(x=%next_input_id_length, name="51_item")
# %concat_41: (4,int32)^(Tensor) = concat(values=(%concat_41_values0_0, %concat_41_values1_0, %51_item, %concat_41_values3_0), axis=0, interleave=False, name="concat_41")
# %new_kv_cache: (is95, 1, is96, is97, fp16)(Tensor) = slice_by_index(x=%new_kv_cache.1_cast, begin=%concat_41, end=[12, 1, -127, 1536], end_mask=[True, True, False, True], name="1493_cast")