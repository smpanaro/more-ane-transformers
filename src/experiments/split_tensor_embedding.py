import coremltools as ct
import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from os_signpost import Signposter
import os
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

"""
The nn.Embedding layer in GPT2 doesn't run on the Neural Engine, most
likely for the same reason that the output Linear layer doesn't -- tensor
dimension > 16384.

Try an approach to perform an equivalent operation on the Neural Engine.

Result: It seems like the constraint is not tensor dimension. Maybe the
gather op is just not supported on ANE?
"""

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.emb = nn.Embedding(50257, 512)
        self.emb = nn.Embedding(8200, 512)
        self.emb_sel_weights = nn.Parameter(self.emb.weight.unsqueeze(0).unsqueeze(-2))
        self.size = 14000
        # self.emb = nn.Embedding(20, 8)
        # self.size = 3

        def emb_with_weight(weight):
            e = nn.Embedding(*weight.shape)
            e.weight = nn.Parameter(weight)
            return e

        self.emb_splits = nn.ModuleList([
            emb_with_weight(x) for x in
            self.emb.weight.split(self.size)
        ])


    def forward(self, x):
        """
        x: [1, n]
        """
        # x, _ = x.split(1, dim=-1)
        # return self.split_compute_embedding(x)
        return self.emb(x)
        # return torch.index_select(self.emb_sel_weights, 1, x)

    def test_equal(self, x):
        ref = self.emb(x)
        test = self.split_compute_embedding(x)
        assert torch.equal(test, ref)

    def split_compute_embedding(self, x):
        size = self.size
        splits = self.emb_splits
        res = []
        for i, split in enumerate(splits):
            lower, upper = size*i, size*(i+1)-1
            clipped = x #(x - lower).clamp(0, split.weight.shape[0]-1)
            r = split(clipped)
            # print(r.shape, x.shape)
            # r[(x < lower).unsqueeze(-1).expand((*x.shape, split.weight.shape[1]))] = 0 # coremltools does not like this....
            # r[(x > upper).unsqueeze(-1).expand((*x.shape, split.weight.shape[1]))] = 0
            res.append(r)
        return torch.stack(res).sum(dim=0)
        # s = res[0]
        # for r in res[1:]:
        #     s = s + r
        # return s

lookup = torch.randn(8000, 512).view(8000, 512)

@mb.program(input_specs=[mb.TensorSpec(shape=(1,2), dtype=types.int32),],opset_version=ct.target.iOS16)
def custom_gather(x):
    # y = mb.gather_nd(x=lookup.numpy(), indices=x, name="y")
    y = mb.gather(x=lookup.half().numpy(), indices=x, axis=0, batch_dims=0, name="y")
    return y


if __name__ == "__main__":
    # unilm starts (1,128,1,3) that is (n,k,h,w) or (b,c,1,s)
    # torch is (b,s,c)
    x = torch.arange(128).view(1,128)#.unsqueeze(-1)
    # x = torch.arange(20)
    net = Net()

    # net.test_equal(x)


    traced = torch.jit.trace(net, (x,))

    args = {
        "inputs":[
            ct.TensorType("x", shape=ct.Shape(shape=x.shape), dtype=np.int32),
        ],
        "outputs": [
            ct.TensorType("y", dtype=np.float16),
        ],
        "minimum_deployment_target": ct.target.iOS16,
        # "pass_pipeline": ct.PassPipeline.EMPTY,
        # "compute_units": ct.ComputeUnit.CPU_AND_NE,
    }
    mlprog = ct.convert(
        traced,
        convert_to="milinternal",
        **args,
    )
    print(mlprog)

    mlpackage = ct.convert(mlprog, convert_to="mlprogram", **args)
    # mlpackage.save("split_tensor_embedding.mlpackage")

    mlpackage.predict({"x": x.int().numpy()})

# Current
    # {
    #   "nB" : 2,
    #   "top" : "y_cast",
    #   "has_biases" : 0,
    #   "nd_mode" : true,
    #   "nC" : 5,
    #   "blob_weights" : 1,
    #   "weights" : {

    #   },
    #   "is_lookup" : 1,
    #   "type" : "inner_product",
    #   "has_relu" : 0,
    #   "bottom" : "y_cast_expand_dims",
    #   "debug_info" : "y_cast",
    #   "has_tanh" : 0,
    #   "hint_fallback_from_metal" : 1,
    #   "name" : "y_cast",
    #   "has_prelu" : 0
    # },

# Goal
    # {
    #   "nB": 15000,
    #   "top": "60",
    #   "has_biases": 0,
    #   "weights": {
    #     "Q": 1,
    #     "Qscale_t": 3,
    #     "W_t_int8": 5
    #   },
    #   "nC": 512,
    #   "is_lookup": 1,
    #   "quantization_mode": 2,
    #   "type": "inner_product",
    #   "has_relu": 0,
    #   "bottom": "24_0",
    #   "debug_info": "",
    #   "has_tanh": 0,
    #   "nd_mode": true,
    #   "name": "inner_product_0",
    #   "has_prelu": 0
    # },

# Input: 1,128,1,1
    # "24_0" : {
    #   "k" : 128,
    #   "w" : 1,
    #   "n" : 1,
    #   "_rank" : 4,
    #   "h" : 1
    # },
# Output: 1,128,1,512
    # "60" : {
    #   "k" : 128,
    #   "w" : 512,
    #   "n" : 1,
    #   "_rank" : 4,
    #   "h" : 1
    # },