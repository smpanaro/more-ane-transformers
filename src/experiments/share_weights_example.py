import coremltools as ct
import torch
from torch import nn
import numpy as np

"""
Example model to show a case where de-duplicating weights would be useful.
"""

class Net(nn.Module):
    def __init__(self, should_double):
        super().__init__()
        self.should_double = should_double
        self.lin = nn.Linear(3,4)
        torch.nn.init.constant_(self.lin.weight, 1.1)
        torch.nn.init.constant_(self.lin.bias, 0.01)

    def forward(self, x):
        if self.should_double:
            x = x + x
        return self.lin(x)

class BranchNet(nn.Module):
    def __init__(self):
        super().__init__()
        example_input = torch.zeros((3,))
        self.double_net = torch.jit.trace(Net(should_double=True).eval(), (example_input,))
        self.no_double_net = torch.jit.trace(Net(should_double=False).eval(), (example_input,))

    def forward(self, x, cond):
        if cond == 0:
            y = self.no_double_net(x)
        else:
            y = self.double_net(x)
        return y

x = torch.arange(0,3).float()

with torch.no_grad():
    print(f"Net(), should_double=False: {Net(should_double=False).eval()(x)}")
    print(f"Net(), should_double=True: {Net(should_double=True).eval()(x)}")
    print()

    branch_net = BranchNet().eval()
    print(f"BranchNet(), cond==0: {branch_net(x, torch.zeros((1,)))}")
    print(f"BranchNet(), cond!=0: {branch_net(x, torch.ones((1,)))}")
    print()

    print("branch_net.double_net trace code:")
    print(branch_net.double_net.code)

    print("branch_net.no_double_net trace code:")
    print(branch_net.no_double_net.code)

    scripted = torch.jit.script(branch_net)
    print("branch_net scripted code:")
    print(scripted.code)

mlprog = ct.convert(scripted,
                    inputs=[
                        ct.TensorType("x", shape=[3], dtype=np.float32),
                        ct.TensorType("cond", shape=[1], dtype=np.float32)],
                    outputs=[ct.TensorType("y", dtype=np.float32)],
                    compute_precision=ct.precision.FLOAT32,
                    convert_to="milinternal")
print("milinternal program:")
print(mlprog)

mlmodel = ct.convert(mlprog,
                    compute_precision=ct.precision.FLOAT32,
                    convert_to="mlprogram")

print(f'branch_net mlmodel, cond==0: {mlmodel.predict({"x": x.numpy(), "cond": torch.zeros((1,))})["y"]}')
print(f'branch_net mlmodel, cond!=0: {mlmodel.predict({"x": x.numpy(), "cond": torch.ones((1,))})["y"]}')

mlmodel.save("branch-net.mlpackage")