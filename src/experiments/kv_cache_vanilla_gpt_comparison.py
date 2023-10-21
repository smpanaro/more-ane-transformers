import coremltools as ct
import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from os_signpost import Signposter
import os
from src.ml_ane_transformers.vanilla_gpt2 import GPT as VanillaGPT
from models.gpt2 import GPT as KVCacheGPT
from src.utils.psnr import compute_psnr

"""
Compare the PSNR of an unmodified GPT2 (nanoGPT) with the KV cache variant.

Specifically compares the PyTorch versions. See `check_psnr` to compare PyTorch<>CoreML.

Result: PSNR is >100, usually 120-150, until we get to the end of the pad_to_length (512)
after which it falls off because the KV cache model starts dropping data.
"""

kv_model = KVCacheGPT.from_pretrained("gpt2")
vanilla_model = VanillaGPT.from_pretrained("gpt2")

@torch.no_grad()
def predict_kv(model, input_ids, kv_cache=None):
    kv_inputs = model.build_inputs(input_ids, pad_to_length=512, pad_token_id=0)
    kv_input_ids = torch.from_numpy(kv_inputs['input_ids'])
    pos_offset = torch.from_numpy(kv_inputs['pos_offset'])
    qk_mask = torch.from_numpy(kv_inputs['qk_mask'])
    if kv_cache is None:
        kv_cache = model.sample_inputs()['kv_cache']
    return model(kv_input_ids, pos_offset, kv_cache, qk_mask)

@torch.no_grad()
def predict_vanilla(model, input_ids):
    x = input_ids if input_ids.shape[-1] <= 512 else input_ids[:, -512:]
    return model(x)

input_ids = torch.randint(10_000, (1,126,))

kv_logit_history = []
vanilla_logit_history = []

kv_cache = None
for i in range(20):
    kv_out, kv_cache = predict_kv(kv_model, input_ids, kv_cache)
    vanilla_out, _ = predict_vanilla(vanilla_model, input_ids)

    kv_logit_history.append(kv_out)
    vanilla_logit_history.append(vanilla_out)

    assert kv_cache is not None
    assert kv_out.shape == vanilla_out.shape, f"{kv_out.shape} != {vanilla_out.shape}"
    assert kv_out.dtype == vanilla_out.dtype, f"{kv_out.dtype} != {vanilla_out.dtype}"

    psnr = compute_psnr(vanilla_out, kv_out)
    print("PSNR:", psnr, i+126)

    next_token = vanilla_out.argmax(dim=-1)
    input_ids = torch.cat([input_ids, next_token], dim=1) # rolling window
