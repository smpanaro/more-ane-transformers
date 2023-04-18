import coremltools as ct
import torch
from torch import nn
import numpy as np
from stopwatch import Stopwatch

"""
See if there are any performance optimizations for matrices with a lot of zeros.

Answer: Doesn't seem like it.
Also, using cond forces you off the Neural Engine so any switching deep in the model is not going to work. (For KV-cache related branching.)
"""

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_up = nn.Linear(512, 1024)
        self.linear_down = nn.Linear(1024, 512)

    def forward(self, x, B, mask):
        """
        x: [512, 512]
        """

        @torch.jit.script_if_tracing
        def matmul(x, B, mask):
            if mask > 0:
                newx = x @ B
            else:
                newx = x
                x_other = x[mask, :] @ B
                newx[mask, :] = x_other
            return newx

        for _ in range(10):
            x = self.linear_up(x)
            # x = x @ B
            x = matmul(x, B, mask)
            x = self.linear_down(x)
        return x

net = Net().eval()

traced = torch.jit.trace(net, (torch.randn((512, 512)), torch.randn(1024, 1024), torch.tensor([1])))
print(traced.code)
mlmodel = ct.convert(traced,
    inputs=[
        ct.TensorType("x", shape=(512, 512), dtype=np.float32),
        ct.TensorType("B", shape=(1024, 1024), dtype=np.float32),
        ct.TensorType("mask", shape=[1], dtype=np.int32),
    ],
    outputs=[
        ct.TensorType("x_out", dtype=np.float32),
    ],
    compute_precision=ct.precision.FLOAT32,
    minimum_deployment_target=ct.target.iOS16,
    convert_to="milinternal"
)
print(mlmodel)
mlmodel = ct.convert(mlmodel,
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS16,
    compute_units=ct.ComputeUnit.CPU_ONLY,
    convert_to="mlprogram"
)

nonzero = torch.randn((512, 512)).float()
zeroed = torch.zeros((512, 512)).float()
zeroed[120, :] = nonzero[120, :]
mask = torch.tensor([120]).int()
fake_mask = torch.tensor([-1]).int()
B = torch.randn((1024, 1024))

sw = Stopwatch(3)
for _ in range(10):
    mlmodel.predict({"x": nonzero.numpy(), "B": B.numpy(), "mask": fake_mask.numpy()})
sw.stop()
print(f"nonzero total: {sw}")

sw.reset()
sw.start()
for _ in range(10):
    mlmodel.predict({"x": zeroed.numpy(), "B": B.numpy(), "mask": mask.numpy()})
sw.stop()
print(f"zeroed total: {sw}")


mlmodel.predict({"x": nonzero.numpy(), "B": B.numpy(), "mask": fake_mask.numpy()})