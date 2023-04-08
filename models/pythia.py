"""
nanoGPT + huggingface inspired-implementation of the pythia models from EleutherAI.
This includes neox.

Please read the 'Out-of-scope use' section on the Pythia huggingface pages. Notably:
"The Pythia Suite is not intended for deployment. It is not a in itself a product and
cannot be used for human-facing interactions."
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.n_head == 0
        self.n_head = config.n_head
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.n_head
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        # max_positions = config.max_position_embeddings
        # self.register_buffer(
        #     "bias",
        #     torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
        #         1, 1, max_positions, max_positions
        #     ),
        # )
        # self.register_buffer("masked_bias", torch.tensor(-1e9))
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims, config.max_position_embeddings, base=config.rotary_emb_base
        )
        self.norm_factor = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype())
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    # def forward(self, x, position_ids, attention_mask):
    #     B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

    #     # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    #     q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
    #     k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
    #     q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
    #     v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)

    #     # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    #     # manual implementation of attention
    #     att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    #     # ANE: Using a stored bias makes the model file larger (from a little to a lot)
    #     # because it is copied into the model proto for every usage. Additionally, mask
    #     # fill only runs on the CPU and moving from ANE <-> CPU is sloooow.
    #     # Instead follow the approach from ml-ane-transformers, subtract a large but
    #     # float16-friendly value so that masked values are effectively ignored in the softmax.
    #     # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-1e4'))
    #     att = att + attention_mask
    #     att = F.softmax(att, dim=-1)
    #     att = self.attn_dropout(att)
    #     y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    #     y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

    #     # output projection
    #     y = self.resid_dropout(self.c_proj(y))
    #     return y

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor

    def forward(self, x, position_ids, attention_mask):
        # assert x.shape == torch.Size([1, 11,512]), x.shape
        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(x)
        # assert qkv.shape == torch.Size([1, 11, 512*3]), qkv.shape

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.n_head, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)
        # assert qkv.shape == torch.Size([1, 11, 8, (512//8)*3]), qkv.shape

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Compute attention
        attn_output, _ = self._attn(query, key, value, attention_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.n_head, self.head_size)
        attn_output = self.dense(attn_output)

        return attn_output

    def _attn(self, query, key, value, attention_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor),
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        # mask_value = torch.finfo(attn_scores.dtype).min
        # # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        # mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        # attn_scores = torch.where(causal_mask, attn_scores, mask_value)
        # FIXME: Add attention_mask

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        # if head_mask is not None:
        #     attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        # x: [bs, n_head, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:seq_len, ...].to(x.device), self.sin_cached[:seq_len, ...].to(x.device)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU() # pretty sure this is the default: approximate=False

    def forward(self, x):
        x = self.dense_h_to_4h(x)
        x = self.act(x)
        x = self.dense_4h_to_h(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, position_ids, attention_mask=None):
        # Computed in parallel.
        attn_out = self.attention(self.input_layernorm(x), position_ids, attention_mask)
        mlp_output = self.mlp(self.post_attention_layernorm(x))
        x = mlp_output + attn_out + x # the order matters...
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    hidden_size: int = 512
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    layer_norm_eps: float = 1e-05
    rotary_pct: float = 0.25
    rotary_emb_base: int = 10000

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, output_mask=None):
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        position_ids = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # ANE: Since we are only inferring and we only care about predicting the next token,
        # we can use the same triangular mask always. May need to change this to support flexible sizes.
        attention_mask = (1 - torch.tril(torch.ones((1,1,t,t), dtype=torch.float16))) * -1e4

        inputs_embeds = self.embed_in(input_ids)
        x = inputs_embeds

        # forward the GPT model itself
        for l in self.layers:
            x = l(x, position_ids, attention_mask)

        # No need to compute anything else for the other length-1 tokens.
        # This shaves ~30% off of gpt2-xl when running on CPU+ANE.
        if output_mask is not None:
            x = torch.index_select(x, 1, output_mask)

        x = self.final_layer_norm(x)

        logits = self.lm_head(x)

        return logits

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        model_type = model_type.replace('EleutherAI/', '')
        assert model_type in {'pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-2.8b', 'pythia-6.9b'}
        model_type = 'EleutherAI/' + model_type
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPTNeoXForCausalLM
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and hidden_size are determined from model_type
        config_args = {
            'pythia-70m':      dict(n_layer=6,  n_head=8,  hidden_size=512,  intermediate_size=2048),
            'pythia-160m':     dict(n_layer=12, n_head=12, hidden_size=768,  intermediate_size=3072),
            'pythia-410m':     dict(n_layer=24, n_head=16, hidden_size=1024, intermediate_size=4096),
            'pythia-2.8b':     dict(n_layer=32, n_head=32, hidden_size=2560, intermediate_size=10240),
            'pythia-6.9b':     dict(n_layer=32, n_head=32, hidden_size=4096, intermediate_size=16384, vocab_size=50432),
        }[model_type.replace('EleutherAI/', '')]
        # config_args['vocab_size'] = 50432 if model_type in ['pythia-6.9b'] else 50304 # always 50304 for GPTNeoX model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints #FIXME
        config_args['bias'] = True # always True for GPT model checkpoints # FIXME
        print(f"forcing vocab_size={config_args['vocab_size']}, block_size={config_args['block_size']}, bias={config_args['bias']}")
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.foo')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPTNeoXForCausalLM.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        for k in list(sd_hf.keys()):
            if k.startswith("gpt_neox."):
                newk = k.replace("gpt_neox.", "")
                assert newk not in sd_hf
                sd_hf[newk] = sd_hf.pop(k)

        for k in list(sd_hf.keys()):
            if k.startswith("embed_out."):
                newk = k.replace("embed_out.", "lm_head.")
                sd_hf[newk] = sd_hf.pop(k)
            elif k.endswith(".masked_bias"):
                del sd_hf[k]
            elif k.endswith(".attention.bias"): #FIXME what is this
                del sd_hf[k]

        in_hf = set(sd_hf.keys()) - set(sd_keys)
        missing_hf = set(sd_keys) - set(sd_hf.keys())
        if len(in_hf) != 0 or len(missing_hf) != 0:
            print([x for x in in_hf if "" in x])
            print([x for x in missing_hf if "" in x])

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            assert sd_hf[k].shape == sd[k].shape, f"{k}: {sd_hf[k].shape} != {sd[k].shape}"
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

        return model

if __name__ == "__main__":
    import argparse
    from transformers import GPTNeoXForCausalLM, AutoTokenizer
    from src.utils.psnr import compute_psnr
    parser = argparse.ArgumentParser(description='Convert a model to CoreML.')
    parser.add_argument('--model_name', choices=['pythia-70m', 'pythia-160m'], default="pythia-70m", type=str)
    args = parser.parse_args()

    model_name = 'EleutherAI/' + args.model_name

    nano = GPT.from_pretrained(model_name).eval()
    hf = GPTNeoXForCausalLM.from_pretrained(model_name, use_cache=False).eval()

    tok = AutoTokenizer.from_pretrained(model_name)

    inputs = tok("this washed coffee comes from huila, colombia", return_tensors="pt")
    with torch.no_grad():
        # print(inputs)
        # inputs = torch.rand((1,1,512), requires_grad=False)
        # position_ids = torch.arange(0, inputs.shape[1], dtype=torch.long).unsqueeze(0) # shape (1, t)
        # attention_mask = (1 - torch.tril(torch.ones((1,1,inputs.shape[1],inputs.shape[1]), dtype=torch.float16))) * -1e4
        # attention_mask = (1 - torch.tril(torch.ones((1,1,inputs.shape[1],inputs.shape[1]), dtype=torch.float32))) * -1e9
        # hf_out_a = hf.gpt_neox.layers[0].attention(hf.gpt_neox.layers[0].input_layernorm(inputs), None)[0]
        # nano_out_a = nano.layers[0].attention(nano.layers[0].input_layernorm(inputs), position_ids, attention_mask)
        # assert torch.equal(hf_out_a, nano_out_a), "zzz"
        # hf_out_mlp = hf.gpt_neox.layers[0].mlp(hf.gpt_neox.layers[0].post_attention_layernorm(inputs))
        # nano_out_mlp = nano.layers[0].mlp(nano.layers[0].post_attention_layernorm(inputs))
        # assert torch.equal(hf_out_mlp, nano_out_mlp), "lkjlkjlkj"
        # hf_out = hf_out_mlp + hf_out_a + inputs
        # nano_out = nano_out_mlp + nano_out_a + inputs
        # assert torch.equal(hf_out, nano_out), "wqwwqwqwq"

        # hf_out_l = hf.gpt_neox.layers[0](inputs, None)[0]
        # nano_out_l = nano.layers[0](inputs, position_ids, attention_mask)
        # wtf = nano_out_a + nano_out_mlp + inputs
        # print(wtf - nano_out_l)
        # assert torch.equal(hf_out_l, hf_out)
        # assert torch.equal(wtf, nano_out_l)
        # assert torch.equal(hf_out_l, nano_out_l)


        inputs = {k:v for k,v in inputs.items() if k in ["input_ids"]}
        hf_out = hf(**inputs)['logits']
        nano_out = nano(**inputs)

    assert hf_out.shape == nano_out.shape, f"{hf_out.shape} != {nano_out.shape}"
    # psnr should be ~240 if perfect.
    print("psnr:", compute_psnr(hf_out, nano_out))
    print("eq", torch.equal(hf_out, nano_out))
