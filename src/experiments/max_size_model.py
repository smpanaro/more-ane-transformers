import torch
from torch import nn
import numpy as np
import coremltools as ct
import sys

"""
Generate large models (either number of parameters or number of ops) to
try and find the limits of the neural engine. It seems to be size-based.
"""

class Net(nn.Module):
    def __init__(self, num_loops):
        super().__init__()
        # Many weights.
        self.ls = nn.ModuleList([
            l for _ in range(num_loops)
            for l in [nn.Linear(1600, 6400), nn.GELU(), nn.Linear(6400, 1600)]
        ])
        # Many ops.
        # self.ls = nn.ModuleList([
        #     l for _ in range(num_loops)
        #     for l in [nn.Linear(20, 10), nn.GELU(), nn.Linear(10, 20)]
        # ])

    def forward(self, x):
        for l in self.ls:
            x = l(x)

        return x

if __name__ == "__main__":
    size = 20 # 20 gets you about 800MB.
    net = Net(size).eval()

    input_ids = torch.rand((1,512,1600,), dtype=torch.float32)

    if input_ids.shape[-1] > 20 and size > 200:
        print("Comment this out if you really want to make a 10GB+ model.")
        sys.exit(1)

    traced = torch.jit.trace(net, (input_ids))

    total_params = sum(p.numel() for p in traced.parameters())
    print(f"{total_params} params")
    print(f"{(total_params * 16 / 8) / 1024 / 1024} MB @ f16")

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
        ],
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    mlmodel.save(f"test-net-{size}-loops.mlpackage")