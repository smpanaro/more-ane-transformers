import torch
from torch import nn

EMBED_DIM = 768

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

c_proj = Conv1D(EMBED_DIM, EMBED_DIM)
#c_proj = nn.Conv1d(EMBED_DIM, EMBED_DIM, 1)
print("1d weight shape:", c_proj.weight.shape)
print("1d bias shape:", c_proj.bias.shape)

out_proj = nn.Conv2d(EMBED_DIM, EMBED_DIM, 1)
print("2d weight shape:", out_proj.weight.shape)
print("2d bias shape:", out_proj.bias.shape)

with torch.no_grad():
    out_proj.weight.copy_(c_proj.weight.t().unsqueeze(-1).unsqueeze(-1))
    out_proj.bias.copy_(c_proj.bias)

t = torch.rand(EMBED_DIM, EMBED_DIM)
oned_out = c_proj(t)
twod_out = out_proj(t.unsqueeze(-1).unsqueeze(-1))
print("1d out", oned_out.shape)
print("2d out", twod_out.shape)

twod_out = twod_out.squeeze(-1).squeeze(-1)
print(nn.CosineSimilarity(dim=-1)(oned_out, twod_out).mean())
print(torch.equal(oned_out, twod_out))
