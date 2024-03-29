"""
A fairly modified version of nanoGPT that attempts to apply the principles
from the ml-ane-transformers repo. Functional, but not as fast as I had hoped.

References:
https://github.com/karpathy/nanoGPT/blob/a82b33b525ca9855d705656387698e13eb8e8d4b/model.py#L1
https://github.com/apple/ml-ane-transformers/tree/main/ane_transformers/reference

Original nanoGPT License:
MIT License

Copyright (c) 2022 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from .ane.multihead_attention import SelfAttention as AneSelfAttention
# from ane.multihead_attention import MultiHeadAttention as AneMultiHeadAttention
from .ane.layer_norm import LayerNormANE as AneLayerNormx
from .ane.dummy_layer_norm import DummyLayerNormANE
from .ane.kahan_layer_norm import KahanLayerNormANE
from .ane.ffn import FFN as AneMLP

# Torch does not support layer norm over the 1st dimension. But the MIL (almost) does.
# Override this to use the native 1st dimension MIL layer norm.
# See: https://github.com/apple/coremltools/pull/1972 to avoid this.
OVERRIDE_LAYER_NORM = False
layer_norm_cls = DummyLayerNormANE if OVERRIDE_LAYER_NORM else AneLayerNormx

from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

if OVERRIDE_LAYER_NORM:
    if "batch_norm" in _TORCH_OPS_REGISTRY:
        del _TORCH_OPS_REGISTRY["batch_norm"]
    @register_torch_op
    def batch_norm(context, node):
        inputs = _get_inputs(context, node, expected=9)
        _input = inputs[0]
        weight = inputs[1]
        bias = inputs[2]
        # running_mean = inputs[3]
        # running_var = inputs[4]
        # training = inputs[5].val
        eps = inputs[7]

        # Ideally would not need the transposes, but axes=[1] doesn't work
        # on the Neural Engine. https://developer.apple.com/forums/thread/728931
        # Edit: Actually axes=[1] works, but it overflows sooner than axes=[3] on Venutura.

        # node.name has to go on the last op, that's also the one that gets added to the context
        rs1 = mb.transpose(x=_input, perm=[0,3,2,1])

        ln = mb.layer_norm(x=rs1, axes=[3], epsilon=eps, gamma=weight, beta=bias)
        # context.add(ln)

        rs2 = mb.transpose(x=ln, perm=[0,3,2,1], name=node.name)
        context.add(rs2)

# Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
# apply scale and bias terms in opposite orders. In order to accurately restore a
# state_dict trained using the former into the the latter, we adjust the bias term
def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata,
                                           strict, missing_keys,
                                           unexpected_keys, error_msgs):
    if not OVERRIDE_LAYER_NORM:
        state_dict[prefix +'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix +'weight']
    return state_dict


# class AneLayerNorm(AneLayerNormx):
class AneLayerNorm(layer_norm_cls):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(
            correct_for_bias_scale_order_inversion)

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_1 = AneLayerNorm(config.n_embd)
        # self.attn = CausalSelfAttention(config)
        self.attn = AneSelfAttention(embed_dim=config.n_embd, n_head=config.n_head, dropout=config.dropout, return_weights=False)
        # self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = AneLayerNorm(config.n_embd)
        self.mlp = AneMLP(config.n_embd, 4 * config.n_embd, dropout=config.dropout)

    def forward(self, x, qk_mask=None, k_mask=None):
        """
        x: (batch, embed_dim, 1, seqlen) aka BC1S
        qk_mask, k_mask as in AneSelfAttention
        """
        # print("xin", x.shape)
        # print("self.ln_1(x)", self.ln_1(x).shape)
        # print(" self.attn(self.ln_1(x))",  self.attn(self.ln_1(x)).shape)
        x = x + self.attn(self.ln_1(x), qk_mask=qk_mask, k_mask=k_mask)
        x = x + self.mlp(self.ln_2(x))
        # print("xout", x.shape)
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ln_f = AneLayerNorm(config.n_embd),
        ))
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head = nn.Conv2d(config.n_embd, config.vocab_size, 1, bias=False) # Confirmed that this is equivalent to ^. See: https://sebastianraschka.com/faq/docs/fc-to-conv.html
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # FIXME: This duplicates these (large) weights.
        self.transformer.wte.weight = nn.Parameter(self.lm_head.weight.squeeze()) # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        self.qk_mask = ((1 - torch.tril(torch.ones((512,512), dtype=torch.float32))).t() * -1e4).view(1, 512, 1, 512)

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
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def prepare_for_block(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # print("x pre-blocks", x.shape)
        # BSC -> BC1S (most conducive to ANE)
        # Batch=1, Sequence=NumTokens, Channels=NumEmbed aka Hidden Size
        return x.transpose(1, 2).unsqueeze(2)

    def forward(self, idx, output_mask=None, qk_mask=None, k_mask=None, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        if qk_mask is None:
            # qk_mask = ((1 - torch.tril(torch.ones((512,512), dtype=torch.float32))).t() * -1e4).view(1, 512, 1, 512)
            qk_mask = self.qk_mask[:, :t, :, :t].to(device)

        # print("forwardz", idx.shape, self.transformer.wte.weight.shape)
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # print("x pre-blocks", x.shape)
        # BSC -> BC1S (most conducive to ANE)
        # Batch=1, Sequence=NumTokens, Channels=NumEmbed aka Hidden Size
        x = x.transpose(1, 2).unsqueeze(2)

        for block in self.transformer.h:
            x = block(x, qk_mask=qk_mask, k_mask=k_mask)
        x = self.transformer.ln_f(x)

        # No need to pass back the other length-1 results we don't care about.
        # Big speed boost.
        if output_mask is not None:
            x = torch.index_select(x, 3, output_mask)
            # TODO: Sprinkle in a softmax here? Or does that prevent us from doing top-p/top-k
            loss = None

        # logits = x
        old = False
        if old:
            # Old slow way.
            x = x.permute(3, 1, 0, 2) # (S, C, B, 1)
            logits = self.lm_head(x).squeeze().unsqueeze(0).unsqueeze(0)
        else:
            # # New fast way.
            # This part is actually quite fast. 4ms out of 75ms.
            x = x.permute(0,2,3,1).squeeze(0)

            # ANE can only load tensors with dim size of at most 16,384 - divide the vocab size to be less than that
            # gpt2 vocab size is a product of two primes smh
            splits = self.transformer.wte.weight.split(self.transformer.wte.weight.shape[0]//29, dim=0)
            logits = torch.cat([torch.einsum('bid,jd->bij', x, split) for split in splits], dim=-1)#.view(*x.shape[:2], -1)

        return logits

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'ckiplab/gpt2-tiny-chinese'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'ckiplab/gpt2-tiny-chinese': dict(n_layer=4, n_head=12, n_embd=312), # 18M, vocab size 21128
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 21128 if "chinese" in model_type else 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        if OVERRIDE_LAYER_NORM:
            sd_keys = [k for k in sd_keys if not k.endswith("running_mean")]
            sd_keys = [k for k in sd_keys if not k.endswith("running_var")]
            sd_keys = [k for k in sd_keys if not k.endswith("num_batches_tracked")]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        transform_hf_weights(sd_hf)

        # print("ane", sd['transformer.h.0.attn.out_proj.weight'].shape)
        # print("hf", sd_hf['transformer.h.0.attn.c_attn.weight'].shape)

        # HF uses one weights matrix for QKV (called C), but ANE uses 3.
        # TODO: Maybe an opportunity to simplify? IDK
        # for k in list(sd_hf.keys()):
        #     # attention input
        #     if "attn.c_attn" in k:
        #         new_k = lambda nk: k.replace("c_attn", nk)

        #         catt = sd_hf[k]
        #         dim = -1 # bias
        #         if "weight" in k:
        #             # catt = catt.unsqueeze(-1).unsqueeze(-1)
        #             dim = -3 # weight

        #         # FIXME: Not sure about the order.
        #         # From HF: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        #         qt, kt, vt = catt.chunk(3, dim=dim)
        #         print("catt shape", catt.shape)

        #         assert qt.shape == sd[new_k("k_proj")].shape,\
        #             f"{k}: {qt.shape} != {sd[new_k('k_proj')].shape}"
        #         sd_hf[new_k("k_proj")] = kt
        #         sd_hf[new_k("q_proj")] = qt
        #         sd_hf[new_k("v_proj")] = vt
        #         del sd_hf[k]
        #     # attention output
        #     elif "attn.c_proj" in k:
        #         new_k = lambda nk: k.replace("c_proj", nk)

        #         catt = sd_hf[k]
        #         # if "weight" in k:
        #         #     catt = catt.unsqueeze(-1).unsqueeze(-1)

        #         new_out = catt
        #         assert new_out.shape == sd[new_k("out_proj")].shape,\
        #             f"{k}: {new_out.shape} != {sd[new_k('out_proj')].shape}"
        #         sd_hf[new_k("out_proj")] = new_out
        #         del sd_hf[k]

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)

        if len(set(sd_keys) ^ set(sd_keys_hf)) > 0:
            print(set(sd_keys_hf) - set(sd_keys))
            print(set(sd_keys) - set(sd_keys_hf))

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            # if any(k.endswith(w) for w in transposed):
            #     print("tpose", k, sd_hf[k].shape)
            #     # special treatment for the Conv1D weights we need to transpose
            #     assert sd_hf[k].shape[::-1] == sd[k].shape, f"{k}: {sd_hf[k].shape[::-1]} == {sd[k].shape}"
            #     with torch.no_grad():
            #         sd[k].copy_(sd_hf[k].t())
            # if any(k.endswith(w) for w in transposed_and_unsqueezed):
            #     print("tpose+unsq", k)
            #     # special treatment for the Conv1D weights we need to transpose and unsqueeze
            #     # :-2 to drop the last 2 singleton dimensions
            #     # ::-1 to transpose
            #     assert sd_hf[k].shape[::-1] == sd[k].shape[:-2], f"{k}: {sd_hf[k].shape[::-1]} == {sd[k].shape[:-2]}"
            #     with torch.no_grad():
            #         sd[k].copy_(sd_hf[k].t().unsqueeze(-1).unsqueeze(-1))
            # else:
                # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape, f"{k}: {sd_hf[k].shape} != {sd[k].shape}"
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @staticmethod
    def build_inputs(seq, pad_to_length=None, pad_token_id=-1, dtype=torch.float32, device="cpu"):
        seqlen = seq.shape[1]

        if not pad_to_length:
            pad_to_length = seqlen
        length = pad_to_length

        assert length == seqlen or pad_token_id != -1, "pad token must be provided when padding"

        # Upper triangular mask but in the BC1S format. aka causal mask
        # Note the transpose! Don't really get it, but easy to see when comparing the results of applying the mask.
        qk_mask = ((1 - torch.tril(torch.ones((length,length), dtype=dtype))).t() * -1e4).view(1, length, 1, length)

        # We want to attend to the whole sequence, but not the padding. aka attention mask
        k_mask = torch.cat([
            torch.zeros(seqlen, dtype=dtype),
            torch.full((pad_to_length - seqlen,), float(-1e4), dtype=dtype)
        ]).view(1,length,1,1)

        # Pad the sequence itself too.
        input_ids = torch.cat([
            seq.squeeze(),
            torch.full((pad_to_length - seqlen,), pad_token_id)
        ]).unsqueeze(0)

        # Used to mask outputs before they exit the model.
        # input_ids: [0,1,2,3] length = 4, result is in index 3
        output_mask = torch.tensor([seqlen-1], dtype=torch.int32)

        return {
            "input_ids": input_ids.int().to(device),
            "qk_mask": qk_mask.to(device),
            "k_mask": k_mask.to(device),
            "output_mask": output_mask.to(device),
        }

# def linear_to_conv2d_map(state_dict):
#     """ Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights
#     """
#     for k in state_dict:
#         print(k, state_dict[k].shape)
#         # is_internal_proj = all(substr in k for substr in ['lin', '.weight'])
#         is_internal_proj = all(substr in k for substr in ['attn', '.weight'])
#         is_output_proj = all(substr in k
#                              for substr in ['classifier', '.weight'])
#         is_lm_head = all(substr in k for substr in['lm_head', '.weight'])
#         if is_internal_proj or is_output_proj or is_lm_head:
#             if len(state_dict[k].shape) == 2:
#                 if "attn.c_proj" in k: # or "attn.c_attn" in k:
#                     print("YOOOOOOOO", k)
#                     state_dict[k] = state_dict[k].t()
#                 state_dict[k] = state_dict[k][:, :, None, None]

def transform_hf_weights(state_dict):
    for k in list(state_dict.keys()):
        # print(k, state_dict[k].shape)

        # In HF defined as:
        # self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        # Here:
        # self.out_proj = nn.Conv2d(self.d_v, self.d_out, 1)
        # To go from Conv1D to Conv2d, we need to transpose and unsqueeze twice.
        if "attn.c_proj.weight" in k:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            newk = k.replace("attn.c_proj.weight", "attn.out_proj.weight")
            state_dict[newk] = state_dict.pop(k).t()[:, :, None, None]
            # after = state_dict[newk].shape
            # print(before, "->", after)

        # Similar to c_proj.weight
        elif "attn.c_proj.bias" in k:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            newk = k.replace("attn.c_proj.bias", "attn.out_proj.bias")
            state_dict[newk] = state_dict.pop(k)
            # after = state_dict[newk].shape
            # print(before, "->", after)

        # In HF defined as:
        # self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        # Here:
        # self.q_proj = nn.Conv2d(embed_dim, self.d_qk, 1)
        # self.v_proj = nn.Conv2d(embed_dim, self.d_v, 1)
        # self.k_proj = nn.Conv2d(embed_dim, self.d_qk, 1)
        # To go from Conv1D to Conv2d, we need to transpose and unsqueeze twice.
        elif "attn.c_attn.weight" in k:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            # From HF: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            qm,km,vm = state_dict.pop(k).t()[:, :, None, None].chunk(3, dim=0)
            state_dict[k.replace("c_attn", "q_proj")] = qm
            state_dict[k.replace("c_attn", "k_proj")] = km
            state_dict[k.replace("c_attn", "v_proj")] = vm
            # newks = [k.replace("c_attn", f"{l}_proj") for l in ["q", "k", "v"]]
            # afters = [state_dict[newk].shape for newk in newks]
            # print(before, "->", afters)

        # Similar to c_attn.weight
        elif "attn.c_attn.bias" in k:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            qm,km,vm = state_dict.pop(k).chunk(3, dim=0)
            state_dict[k.replace("c_attn", "q_proj")] = qm
            state_dict[k.replace("c_attn", "k_proj")] = km
            state_dict[k.replace("c_attn", "v_proj")] = vm
            # newks = [k.replace("c_attn", f"{l}_proj") for l in ["q", "k", "v"]]
            # afters = [state_dict[newk].shape for newk in newks]
            # print(before, "->", afters)

        # In HF defined as:
        # self.c_fc = Conv1D(intermediate_size, embed_dim)
        # Here:
        # self.c_fc = nn.Conv2d(embed_dim, ffn_dim, 1)
        # To go from Conv1D to Conv2d, we need to transpose and unsqueeze twice.
        elif "mlp.c_fc.weight" in k:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            state_dict[k] = state_dict[k].t()[:, :, None, None]
            # after = state_dict[k].shape
            # print(before, "->", after)

        # In HF defined as:
        # self.c_proj = Conv1D(embed_dim, intermediate_size)
        # Here:
        # self.c_proj = nn.Conv2d(ffn_dim, embed_dim, 1)
        # To go from Conv1D to Conv2d, we need to transpose and unsqueeze twice.
        elif "mlp.c_proj.weight" in k:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            state_dict[k] = state_dict[k].t()[:, :, None, None]
            # after = state_dict[k].shape
            # print(before, "->", after)

        # In HF defined as:
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Here:
        # self.lm_head = nn.Conv2d(config.n_embd, config.vocab_size, 1, bias=False)
        # To go from Linear to Conv2d, we need to unsqueeze twice (NO TRANSPOSE).
        elif "lm_head.weight" in k:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            state_dict[k] = state_dict[k][:, :, None, None]
            # after = state_dict[k].shape
            # print(before, "->", after)

        # Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
        # apply scale and bias terms in opposite orders. In order to accurately restore a
        # state_dict trained using the former into the the latter, we adjust the bias term
        elif ".ln_" in k and ".bias" in k and not OVERRIDE_LAYER_NORM:
            # print(k, state_dict[k].shape)
            # before = state_dict[k].shape
            weight_key = k.replace(".bias", ".weight")
            state_dict[k] = state_dict[k] / state_dict[weight_key]
            # after = state_dict[k].shape
            # print(before, "->", after)

        elif ".ln_" in k and OVERRIDE_LAYER_NORM:
            newk = k.replace(".weight", ".bn.weight")
            newk = newk.replace(".bias", ".bn.bias")
            state_dict[newk] = state_dict.pop(k)


        # else:
        #     if "bias" in k:
        #         print(k)
