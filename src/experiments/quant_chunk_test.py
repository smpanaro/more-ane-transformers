import torch
from torch import nn
import coremltools as ct
import numpy as np

"""
Build a test model, quantize it and use it for debugging
chunking.
"""

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ls = nn.ModuleList([
            nn.Linear(1600, 6400),
            nn.GELU(),
            nn.Linear(6400, 1600),
            nn.GELU(),
            nn.Linear(1600, 6400),
            nn.GELU(),
            nn.Linear(6400, 1600),
        ])

    def forward(self, x):
        for l in self.ls:
            x = l(x)
        return x

if __name__ == "__main__":
    model = Net().eval()

    input_ids = torch.rand((1,512,1600,), dtype=torch.float32)
    traced = torch.jit.trace(model, (input_ids))

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

    quantized = ct.compression_utils.palettize_weights(mlmodel, nbits=2, mode="kmeans")
    quantized.save(f"test-quant-net.mlpackage")