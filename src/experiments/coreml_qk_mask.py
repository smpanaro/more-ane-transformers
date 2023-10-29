import coremltools as ct
import torch
from torch import nn
import numpy as np
from  os_signpost import Signposter
signposter = Signposter("com.smpanaro.more-ane-transformers", Signposter.Category.PointsOfInterest)

"""
See if it's possible to make the QK mask in-model (ie. no dynamic shapes).

Result: it is.
"""

class Net(nn.Module):
    def forward(self, length):
        return self.make_qk_mask(64, 512, length)

    @staticmethod
    def make_baseline_qk_mask(seqlen, maxseqlen, currseqlen):
        # Baseline: this is accurate, but doesn't work in CoreML.
        m = (1 - torch.tril(torch.ones((currseqlen, currseqlen), dtype=torch.float32))) * -1e4
        # Crop any rows off that overflow at the top when currseqlen > seqlen.
        m = m[-seqlen:, :]
        # Add the ignored leading rows for Q tokens when currseqlen < seqlen
        m = torch.cat([torch.ones((max(0, seqlen-currseqlen), currseqlen)) * -1e4, m], dim=0)
        # Add the ignored leading columns for K tokens
        m = torch.cat([torch.ones((seqlen, max(0, maxseqlen-currseqlen),)) * -1e4, m], dim=1)
        m = m.unsqueeze(0).unsqueeze(0)
        return m

    def make_qk_mask(self, seqlen, maxseqlen, currseqlen):
        # Q @ K^T
        # Q (query tokens, dim) @ K^T (dim, key tokens)
        # = (query tokens, key tokens)
        # 1 row for every query token, 1 column for every key token.
        # Each entry is the weight that should be given to the relationship between a single Q and K token.
        # We want:
        # - Q should never attend to a K token that comes after it. (This is normal causal attention.)
        # - In some cases (currseqlen < seqlen), we want to ignore some of the leading Q tokens. (Since we pad on the left.)
        # - In some cases (currseqlen < maxseqlen), we want to ignore some of the leading K tokens. (Since
        #   we pass in garbage for those in the KV cache until they are needed.)
        # We want a matrix something like this:
        # [ -1 -1 -1 -1 -1 -1 -1 -1 -1] < ignore the leading Q tokens
        # [ -1 -1 -1 -1 -1 -1 -1 -1 -1] < ignore the leading Q tokens
        # [ -1 -1 -1 -1  0 -1 -1 -1 -1] < triangular since we don't want a Q token to attend to future tokens
        # [ -1 -1 -1 -1  0  0 -1 -1 -1]
        # [ -1 -1 -1 -1  0  0  0 -1 -1]
        # [ -1 -1 -1 -1  0  0  0  0 -1]
        # [ -1 -1 -1 -1  0  0  0  0  0]
        #    ^  ^  ^  ^ -- ignore the leading K tokens

        # Build a triangular matrix starting from the bottom right and extending up and to the left.
        m = (1 - torch.tril(torch.ones((seqlen, maxseqlen), dtype=torch.float16), diagonal=maxseqlen-seqlen)) * -1e4
        # Mask the leading Q tokens when currseqlen < seqlen
        # m[:max(0, seqlen-currseqlen), :] = -1 # Would be simpler, but doesn't work with CoreML.
        all_mask = torch.full_like(m, -1e4, dtype=torch.float16)
        m = torch.where(torch.arange(m.size(0)).unsqueeze(-1) < max(0, seqlen-currseqlen), all_mask, m)
        # # Mask the leading K tokens.
        m = torch.where(torch.arange(m.size(1)) < max(0, maxseqlen-currseqlen), all_mask, m)

        m = m.unsqueeze(0).unsqueeze(0)

        return m

length = torch.tensor([3]).int()

net = Net()
net.eval()

traced = torch.jit.trace(net, (length,))

args = {
    "inputs": [
        ct.TensorType(name="length", shape=[1], dtype=np.int32),
    ],
    "outputs": [
        ct.TensorType(name="mask", dtype=np.float16),
    ],
    "compute_units": ct.ComputeUnit.CPU_AND_NE,
    "minimum_deployment_target": ct.target.iOS16,
}

prog = ct.convert(traced, **args, convert_to="milinternal")
print(prog)
model = ct.convert(prog, **args, convert_to="mlprogram")

out = model.predict({"length": length.int().numpy()})
torch.set_printoptions(linewidth=150)
print(out["mask"])

for i in range(10):
    with signposter.use_interval(f"length={i}"):
        length = torch.tensor([i]).int()
        out = model.predict({"length": length.int().numpy()})
        baseline = net.make_baseline_qk_mask(64, 512, length)
        print(f"i: {i}\n", out["mask"])
        print(baseline.numpy())
        assert torch.equal(torch.from_numpy(out["mask"]).float(), baseline.float()), f"length={i}"