import torch
from torch import nn
import numpy as np
import coremltools as ct
import time

"""
Generate a large-ish model for evaluating the speed of
ct.convert on different versions of coremltools.
"""

class Net(nn.Module):
    def __init__(self, num_loops):
        super().__init__()
        self.ls = nn.ModuleList([
            l for _ in range(num_loops)
            for l in [nn.Linear(1600, 6400), nn.GELU(), nn.Linear(6400, 1600)]
        ])

    def forward(self, x):
        for l in self.ls:
            x = l(x)
        return x

if __name__ == "__main__":
    # 1600*6400*2 = 20.4M params + 40.96MB per loop
    num_loops = 20
    net = Net(num_loops).eval()

    input_ids = torch.rand((1,512,1600,), dtype=torch.float32)
    traced = torch.jit.trace(net, (input_ids))

    total_params = sum(p.numel() for p in traced.parameters())
    print(f"{total_params / 1000 / 1000:0.3f}M params")
    print(f"{(total_params * 16 / 8) / 1024 / 1024:0.1f} MB @ f16")

    start = time.perf_counter()
    ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="inputs", shape=input_ids.shape, dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="outputs", dtype=np.float32),
        ],
        compute_precision=ct.precision.FLOAT16, # FLOAT32 is faster.
        convert_to="mlprogram",
    )
    end = time.perf_counter()
    print(f"convert took: {end - start:0.4f}s")