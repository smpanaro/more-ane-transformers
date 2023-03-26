import torch

"""
Experiment with splitting the input into chunks and computing
the mean on each chunk separately. If overflowing was a problem,
it might help but you lose a lot of precision.
"""

n_embd = num_channels = 768
seqlen = 512
eps = 1e-5

weight = torch.rand(n_embd)
bias = torch.rand(n_embd)

x = torch.rand([1, n_embd, 1, seqlen])

# torch.Size([1, 768, 1, 512]) torch.Size([768]) torch.Size([768])
# x, weight, bias ^
def baseline(inputs):
    input_rank = len(inputs.size())

    assert input_rank == 4
    assert inputs.size(1) == num_channels
    assert inputs.dtype == torch.float16 or inputs.dtype == torch.float32

    # if self.clip_mag is not None:
    #     inputs.clamp_(-self.clip_mag, self.clip_mag)

    channels_mean = inputs.mean(dim=1, keepdims=True) # shape [1,1,1,S]
    zero_mean = inputs - channels_mean # shape [1,C,1,S]
    zero_mean_sq = zero_mean * zero_mean # shape [1,C,1,S]
    denom = (zero_mean_sq.mean(dim=1, keepdims=True) + eps).rsqrt() # [1,1,1,S]
    out = zero_mean * denom # shape [1,C,1,S]
    out = (out + bias.view(1, num_channels, 1, 1)
            ) * weight.view(1, num_channels, 1, 1)
    return out

def chunked_mean(inputs, num_chunks):
    """
    Compute a mean in chunks to avoid overflows while summing. Assumes BC1S format.
    NOTE: This causes a large loss of precision in float16. Not worth it as far as I can tell.
    """
    num_channels = inputs.shape[1]
    assert num_chunks > 0, "num_chunks must be positive"
    assert num_channels % num_chunks == 0, "num_chunks must evenly divide num_channels"

    if num_chunks == 1:
        return inputs.mean(dim=1, keepdim=True)

    channel_chunks = inputs.split(num_channels//num_chunks, dim=1)
    mean = 0
    for chunk in channel_chunks:
        chunk_sum = chunk.sum(dim=1, keepdim=True) # shape [1,1,1,S]
        chunk_mean = chunk_sum / num_channels
        mean = mean + chunk_mean
    return mean

def chunked(inputs, num_chunks, scale_factor=1e3):
    channels_mean = chunked_mean(inputs / scale_factor, num_chunks) # shape [1,1,1,S]
    channels_mean *= scale_factor

    zero_mean = inputs - channels_mean # shape [1,C,1,S]
    zero_mean_sq = zero_mean * zero_mean # shape [1,C,1,S]

    mean_zero_mean_sq = chunked_mean(zero_mean_sq / scale_factor, num_chunks) * scale_factor
    denom = (mean_zero_mean_sq + eps).rsqrt() # [1,1,1,S]

    # denom = (chunked_mean(zero_mean_sq, num_chunks) + eps).rsqrt() # [1,1,1,S]
    out = zero_mean * denom # shape [1,C,1,S]
    out = (out + bias.view(1, num_channels, 1, 1)
            ) * weight.view(1, num_channels, 1, 1)
    return out

bl = baseline(x)
ch = chunked(x, 1)
print("baseline:", bl.shape)
print("chunked :", ch.shape)
print("equal?  :", torch.equal(bl, ch))
delta = bl - ch
print("max delta:", delta.max(), "median:", delta.median(), "mean:", delta.mean())
