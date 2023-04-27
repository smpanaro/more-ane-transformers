import torch
from torch import nn
import numpy as np

"""
Figure if/how to replace embedding + lm head layers with conv so we can
get to the preferred ANE shape earlier/faster.

Result is the embedding is definitely not possible and the split einsum a la
Whisper works well enough.
"""

# input BT
# gather weight TC
# gather output BSC
# block input (ANE optimal) BC1S
# output from blocks: BC1S
# ...
# desired output BST

B,C,S,T = 1, 312, 512, 2000
# B,C,S,T = 1, 4, 512, 2000
# B,C,S,T = 1, 4, 6, 10

assert C % 2 == 0
assert T % 2 == 0

torch.manual_seed(42)

conv = nn.Conv2d(C, T, 1, bias=False) # lm_head
emb = nn.Embedding(T, C)
emb.weight = nn.Parameter(conv.weight.squeeze())
lin = nn.Linear(C,T, bias=False)
lin.weight = nn.Parameter(conv.weight.squeeze())

x = torch.randn((B, S, C))
lin_res = lin(x)
x = x.transpose(1, 2).unsqueeze(2)
assert x.shape == torch.Size([B,C,1,S])
# print("lin_res", lin_res)
# conv_res = conv(x)
# print("conv_res.shape", conv_res.shape)
# conv_res = conv_res.permute(0,2,3,1).squeeze(0)
# print("conv_res", conv_res)
# print("lin vs conv", torch.equal(lin_res, conv_res))
# print(lin_res - conv_res)
# conv2_res = conv(x.permute(3,1,0,2)).squeeze().unsqueeze(0)
# print("conv_res2", conv_res)
# print("lin vs conv2", torch.equal(lin_res, conv2_res))
# print(lin_res - conv_res)

assert x.shape == torch.Size([B,C,1,S])
x = x.permute(0,2,3,1).squeeze(0)
assert x.shape == torch.Size([B,S,C])

# ANE can only load tensors with dim size of at most 16,384 - divide the vocab size to be less than that
# gpt2 vocab size is a product of two primes smh
print(lin.weight.shape)
splits = emb.weight.split(emb.weight.shape[0]//2, dim=0)
ecat = torch.cat([torch.einsum('bid,jd->bij', x, split) for split in splits], dim=-1)#.view(*x.shape[:2], -1)
print(ecat.shape)
# ecat = torch.einsum('bid,jd->bij', x, emb.weight)
print(ecat.shape)
print(lin_res - ecat)
print(torch.allclose(lin_res, ecat, atol=1e-6))



# @torch.no_grad()
# def naive_way(x):
#     assert x.shape == torch.Size([B,C,1,S])
#     x = x.permute(3, 1, 0, 2) # shape = (S, C, B, 1)
#     logits = conv(x) # shape = (S, T, B, 1)
#     logits = logits.squeeze().unsqueeze(0) # shape = (B, S, T)
#     return logits

# # no faster than naive way
# @torch.no_grad()
# def post_permute_way(x):
#     assert x.shape == torch.Size([B,C,1,S])
#     logits = conv(x) # shape = (B, T, 1, S)
#     assert logits.shape == torch.Size([B,T,1,S])
#     logits = logits.permute(0,2,3,1).squeeze(1) # shape = (B, S, T)
#      # .permute(0,3,1,2).squeeze(-1)
#     return logits

# x = torch.randn((B, C, 1, S))

# print("naive", naive_way(x).shape)
# print("post_permute", post_permute_way(x).shape)
# # print(naive_way(x))
# # print(post_permute_way(x))
# print(torch.allclose(naive_way(x), post_permute_way(x), atol=1e-6))
# print("default tol:", torch.allclose(naive_way(x), post_permute_way(x)))

# # @torch.no_grad()
# # def emb_way(x):
# #     assert x.shape == torch.Size([B,S])
# #     return emb(x)

# # @torch.no_grad()
# # def conv_emb_way(x):
# #     assert x.shape == torch.Size([B,S])
# #     return conv(x)

# # x = torch.randint(2000, (B, S))
# # print("emb", emb_way(x).shape)
# # print("conv emb", conv_emb_way(x).shape)
# # print(torch.allclose(emb_way(x), conv_emb_way(x), atol=1e-6))

