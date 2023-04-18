"""
Originally this was going to be nanoGPT with the bare minimum modifications required
so that it can be converted to CoreML and compared to the ANE implementation. Turns
out it's quite fast out of the box, so tweaked it further to gain more speed.
From: https://github.com/karpathy/nanoGPT/blob/a82b33b525ca9855d705656387698e13eb8e8d4b/model.py#L1

See "ANE:" comments for interesting things.

Original License:
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
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class CacheConfig:
    kv_cache: torch.Tensor # [num_layers, 1, 2*seqlen, n_embd]
    kv_mask: torch.BoolTensor # for masking oldk/oldv [1, seqlen, n_embd], can't build in the model (see comments in Attention class)
    output_mask: torch.Tensor # [1]
    head_index: int

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

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
        # Don't need this for ANE.
        # if not self.flash:
        #     print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.float16))
            #                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attention_mask, kv_config):
        # kv_cache (B, T*2, C)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)

        if kv_config is not None and kv_config.kv_cache is not None:
            kv_cache = kv_config.kv_cache
            kv_mask = kv_config.kv_mask

            new_q, new_k, new_v = q,k,v # new_q, new_k, new_v = [B, 1, n_embd]
            old_k, old_v = kv_cache.chunk(2, dim=1) # each (B, T, C)
            # replace the row indicated by output_mask with the new values

            # Many ways to overwrite a row in a matrix (old K) with a new value (new K).
            # One of them works on ANE.

            # op not supported
            # k = torch.index_copy(old_k, 1, output_mask, new_k)
            # v = torch.index_copy(old_v, 1, output_mask, new_v)

            # https://yuyangyy.medium.com/understand-torch-scatter-b0fd6275331c
            # oldk.scatter(1, torch.tensor([3]).repeat(4).unsqueeze(0).unsqueeze(0), newk) # something like this
            # scatter doesn't run on ANE though :(
            # idx = output_mask.repeat(old_k.shape[2]).unsqueeze(0).unsqueeze(0)
            # k = old_k.scatter(1, idx, new_k)
            # v = old_v.scatter(1, idx, new_v)

            # Like index_copy but supported by coremltools.
            # Problem is it creates a symbolic variable (the :output_mask slicing) which technically
            # has a different type than the non-cached case, so can't be used in a branched if/else model.
            # Also not sure this is numerically accurate.
            # And also doesn't seem to run on the ANE.
            # k = torch.cat([old_k[:, :output_mask, :], new_k, old_k[:, output_mask+1:, :]], dim=1)
            # v = torch.cat([old_v[:, :output_mask, :], new_v, old_v[:, output_mask+1:, :]], dim=1)

            # Fails in index_put AttributeError: 'NoneType' object has no attribute 'sym_type' (the first colon)
            # k = old_k
            # k[:, output_mask] = new_k
            # v = old_v
            # v[:, output_mask] = new_v

            # Equivalent to cat dim=1, same problems.
            # k = torch.hstack([old_k[:, :output_mask, :], new_k, old_k[:, output_mask+1:, :]])
            # v = torch.hstack([old_v[:, :output_mask, :], new_v, old_v[:, output_mask+1:, :]])

            # Fails in index_put AttributeError: 'NoneType' object has no attribute 'sym_type' (the first colon)
            # This would use scatter_nd under the hood anyways which is not ANE compatible.
            # mask = torch.zeros_like(old_k)
            # mask[:, output_mask, :] = 1
            # k = torch.where(mask.bool(), new_k, old_k)
            # v = torch.where(mask.bool(), new_v, old_v)

            # This works, but you have to pass the mask along (on the plus side, you only compute it once).
            k = torch.where(kv_mask, old_k, new_k)
            v = torch.where(kv_mask, old_v, new_v)
            q = new_q

            B,T,C = k.size()

        current_cache = torch.cat([k,v], dim=1)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, q.size()[1], self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # ANE: Using a stored bias makes the model file larger (from a little to a lot)
            # because it is copied into the model proto for every usage. Additionally, mask
            # fill only runs on the CPU and moving from ANE <-> CPU is sloooow.
            # Instead follow the approach from ml-ane-transformers, subtract a large but
            # float16-friendly value so that masked values are effectively ignored in the softmax.
            # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-1e4'))
            att = att + attention_mask
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, -1, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, current_cache

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
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None, kv_config=None):
        attention_output, new_kv_cache = self.attn(self.ln_1(x), attention_mask, kv_config)
        x = x + attention_output
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache

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
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

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

    def forward(self, idx, output_mask=None, kv_cache=None, kv_mask=None, seqlen=512):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # ANE: Since we are only inferring and we only care about predicting the next token,
        # we can use the same triangular mask always. May need to change this to support flexible sizes.
        attention_mask = (1 - torch.tril(torch.ones((1,1,seqlen,seqlen), dtype=torch.float32))) * -1e4

        # kv_cache [# layers, batch size 1, seqlen * 2 = 512*2, hidden size 768]
        if kv_cache is None:
            kv_cache = [None]*len(self.transformer.h)
        else:
            pos = output_mask.unsqueeze(0)
            attention_mask = torch.index_select(attention_mask, 2, output_mask)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        kv_mask_bool = kv_mask.bool() if kv_mask is not None else None
        new_kv_cache = []
        head_index = 0
        for (block, cache) in zip(self.transformer.h, kv_cache):
            kv_config = CacheConfig(cache, kv_mask_bool, output_mask, head_index)
            x, block_new_kv_cache = block(x, attention_mask, kv_config)
            new_kv_cache.append(block_new_kv_cache)
            head_index += 1

        # No need to compute anything else for the other length-1 tokens.
        # This shaves ~30% off of gpt2-xl when running on CPU+ANE.
        if output_mask is not None and t > 1:
            x = torch.index_select(x, 1, output_mask)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        # [# layers, batch size 1, seqlen * 2 = 512*2, hidden size 768]
        # same as input
        new_kv_cache = torch.stack(new_kv_cache) if len(new_kv_cache) > 0 else None

        return logits, new_kv_cache

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @staticmethod
    def config_args():
        return OrderedDict({
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        })

    @staticmethod
    def model_names():
        return list(GPT.config_args().keys())

    @staticmethod
    def tokenizer_by_name():
        return {n:n for n in GPT.model_names()}

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in GPT.model_names()
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = GPT.config_args()[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
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

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
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

if __name__ == "__main__":
    import numpy as np

    def build_kv_mask(output_mask, seqlen=512, hidden_size=768):
        kv_mask = torch.ones(1, seqlen, hidden_size, dtype=torch.int32)
        kv_mask[:, output_mask, :] = 0
        return kv_mask

    model = GPT.from_pretrained("gpt2").eval()

    # Check numeric accuracy before converting to CoreML. It's not exact, might be a bug.
    # if True:
    if False:
        input_ids = torch.randint(10_000, (1, 10,))

        with torch.no_grad():
            base_prediction, _ = model(input_ids, torch.tensor([4]), seqlen=10)
            print("^base ------ first v")

            out, out_cache = model(input_ids, torch.tensor([3]), seqlen=10)

        print("output cache:", out_cache.shape)

        print("----")

        with torch.no_grad():
            out_new, out_new_cache = model(input_ids[:,[4]], torch.tensor([4]), out_cache, seqlen=10)

        print(base_prediction[:, 0:6, 0:4])
        print(out[:, 0, 0:4])
        print(out_new[:, 0, 0:4])
        print("cache sizes", out_cache.shape, out_new_cache.shape)
        print("eq?", torch.equal(out_new[:, 0, :], base_prediction[:, 0, :])) # why aren't these equal?
        print("close?")
        np.testing.assert_allclose(out_new[:, 0, :], base_prediction[:, 0 ,:])

    # Work out the pattern what inputs to pass when.
    # if True:
    if False:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        seq = tok("would the real slim", return_tensors="pt")["input_ids"]
        input_ids = torch.cat([
            seq.squeeze(),
            torch.full((20 - seq.shape[-1],), tok.eos_token_id)
        ]).unsqueeze(0)

        kvc = None
        outs = []
        with torch.no_grad():
            for i in range(5):
                seqlen = seq.shape[-1]+i
                print("i", i, seqlen)
                inputs = input_ids if kvc is None else input_ids[:, [seqlen-1]]
                output_mask = torch.tensor([seqlen-1])
                kv_mask = build_kv_mask(output_mask, seqlen=20, hidden_size=768)
                out, kvc = model(inputs, output_mask=output_mask, kv_cache=kvc, kv_mask=kv_mask, seqlen=20)
                input_ids[0][seqlen] = out.argmax()
                outs.append(out)

        print(tok.decode(input_ids.squeeze().tolist()))

    # Try to build a branching model that can be converted to CoreML.
    # Doesn't run on the Neural Engine and includes duplicate copies of the weights.
    # Would also be difficult/impossible to split into a pipeline for gpt2-xl.
    if True:
    # if False:
        import coremltools as ct

        class DoubleGPT(nn.Module):
            def __init__(self, model_name, seqlen, num_layers, hidden_size):
                super().__init__()
                self.cached = GPT.from_pretrained(model_name)
                self.full = self.cached
                input_ids = torch.randint(10_000, (1, seqlen,))
                kv_cache = torch.zeros(num_layers, 1, seqlen*2,hidden_size)
                kv_mask = torch.zeros(1, seqlen, hidden_size)
                self.cached = torch.jit.trace(self.cached, (input_ids[:, [3]], torch.tensor([3]), kv_cache, kv_mask))
                self.full = torch.jit.trace(self.full, (input_ids, torch.tensor([3])))

            def forward(self, x, output_mask, kv_cache, kv_mask):
                if kv_cache.min() != 0:
                    ci = torch.index_select(x, 1, output_mask)
                    y, new_cache = self.cached(ci, output_mask, kv_cache, kv_mask)
                else:
                    y, new_cache = self.full(x, output_mask)

                return y, new_cache

        seqlen = 512

        # model_name = "gpt2"
        # num_layers = 12
        # hidden_size = 768

        model_name = "gpt2-medium"
        num_layers = 24
        hidden_size = 1024

        input_ids = torch.randint(10_000, (1, seqlen,))
        # traced_model = torch.jit.trace(model, (input_ids, torch.tensor([3]), torch.zeros(num_layers, 1, seqlen*2,hidden_size)))
        with torch.no_grad():
            double = DoubleGPT(model_name, seqlen, num_layers, hidden_size).eval()
            # double(input_ids, torch.tensor([3]), torch.ones(num_layers, 1, seqlen*2,hidden_size))

            kv_cache_shape = (num_layers, 1, seqlen*2,hidden_size)
            kv_mask = torch.ones(1, seqlen, hidden_size)

            # Trace just the cached model.
            traced_cached_model = torch.jit.trace(double, (input_ids, torch.tensor([3]), torch.ones(kv_cache_shape), kv_mask))

            # Trace just the full model.
            traced_full_model = torch.jit.trace(double, (input_ids, torch.tensor([3]), torch.zeros(kv_cache_shape), kv_mask))

            # Script both models.
            # traced_model = torch.jit.script(double)


        def convert_model(traced_model, name):
            print(traced_model.code)
            prog = ct.convert(traced_model,
                inputs=[
                    ct.TensorType(name="input_ids", shape=[1, seqlen], dtype=np.int32),
                    ct.TensorType(name="output_mask", shape=[1], dtype=np.int32),
                    ct.TensorType(name="kv_cache", shape=[num_layers, 1, seqlen*2, hidden_size], dtype=np.float32),
                    ct.TensorType(name="kv_mask", shape=[1, seqlen, hidden_size], dtype=np.int32),
                ],
                outputs=[
                    ct.TensorType(name="logits", dtype=np.float32),
                    ct.TensorType(name="new_kv_cache", dtype=np.float32),
                ],
                minimum_deployment_target=ct.target.iOS16, # TODO: Is this needed?
                # compute_units=ct.ComputeUnit.CPU_AND_GPU if "full" in name else ct.ComputeUnit.CPU_AND_NE,
                compute_precision=ct.precision.FLOAT32,
                convert_to="milinternal")

            # print(prog)

            mlmodel = ct.convert(prog,
                                minimum_deployment_target=ct.target.iOS16, # TODO: Is this needed?
                                convert_to="mlprogram")

            # print(mlmodel.get_spec().description.input)
            # spec = mlmodel.get_spec()
            # spec.description.input[2].type.isOptional = True
            # mlmodel._spec = spec
            # print(mlmodel.get_spec().description.input)

            # mlmodel.save(f"{name}.mlpackage")
            return mlmodel

        cached_mlmodel = convert_model(traced_cached_model, "gpt2kv-cached")
        full_mlmodel = convert_model(traced_full_model, "gpt2kv-full")
        # convert_model(traced_model, "gpt2kv")

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        seq = tok("would the real slim", return_tensors="pt")["input_ids"]
        input_ids = torch.cat([
            seq.squeeze(),
            torch.full((seqlen - seq.shape[-1],), tok.eos_token_id)
        ]).unsqueeze(0)
        original_inputs = {
            "input_ids": input_ids.int().numpy(),
            "output_mask": torch.tensor([seq.shape[1]-1]).int().numpy(),
            "kv_cache": torch.zeros((num_layers, 1, 2*seqlen, hidden_size)).float().numpy(),
        }
        original_inputs["kv_mask"] = build_kv_mask(original_inputs["output_mask"], seqlen=512, hidden_size=hidden_size)
        original_outputs = full_mlmodel.predict(original_inputs)

        # input("Press Enter to continue.")

        from stopwatch import Stopwatch
        stopwatch = Stopwatch(3)
        stopwatch.stop()
        stopwatch.reset()

        input_stopwatch = Stopwatch(3)
        input_stopwatch.stop()
        input_stopwatch.reset()

        # print("input_ids", input_ids)
        # print("output_mask", original_inputs["output_mask"])
        outputs = original_outputs
        num_inferences = 0
        for i in range(min(seqlen, 200) - seq.shape[1]):
            input_stopwatch.start()
            input_ids[0, seq.shape[1]+i] = outputs["logits"].argmax()
            inputs = {
                "input_ids": input_ids.int().numpy(),
                "output_mask": torch.tensor([seq.shape[1]+i]).int().numpy(),
                "kv_cache": outputs["new_kv_cache"], # already numpy floats (from CoreML)
            }
            inputs["kv_mask"] = build_kv_mask(inputs["output_mask"], seqlen=512, hidden_size=hidden_size) # probably cheating to not include this in timing...
            input_stopwatch.stop()
            # print("input_ids", input_ids)
            # print("output_mask", inputs["output_mask"])

            stopwatch.start()
            outputs = cached_mlmodel.predict(inputs)
            # outputs = full_mlmodel.predict(inputs)
            stopwatch.stop()

            num_inferences += 1

        input_ids[0, seq.shape[1]+i] = outputs["logits"].argmax()
        print("final input_ids", input_ids)
        print(tok.decode(input_ids[0, :]))

        print("\n\n---stats---")
        per_inference = "{:.{}f}ms".format((stopwatch.duration / num_inferences) * 1000, 2)
        print(stopwatch, "total")
        print(f"{per_inference}/it")

        input_per_inference = "{:.{}f}ms".format((input_stopwatch.duration / num_inferences) * 1000, 2)
        print(f"input prep {input_per_inference}/it")


        # print(prog)
