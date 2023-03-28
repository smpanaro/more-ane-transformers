import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import coremltools as ct
import numpy as np
from datetime import datetime
from src.utils.psnr import compute_psnr
from src.ml_ane_transformers.ane_gpt2 import GPT as ANEGPT

"""
Experiment to see if we need k, qk or both when we only care about the
next token prediction.

PSNR between just qk + both is very high, so seems that k is unneeded.
"""

ane = ANEGPT.from_pretrained("gpt2").eval()

random_tokens = torch.randint(10000, (1,10,))
inputs_dict = ane.build_inputs(random_tokens, pad_to_length=512, pad_token_id=350)
input_ids, qk_mask, k_mask, output_mask = [inputs_dict[k] for k in\
                                            ["input_ids", "qk_mask", "k_mask", "output_mask"]]
assert output_mask[0] == 9, f"{output_mask} is not as expected"

with torch.no_grad():
    k_only = ane(input_ids, qk_mask=None, k_mask=k_mask, output_mask=output_mask)
    qk_only = ane(input_ids, qk_mask=qk_mask, k_mask=None, output_mask=output_mask)
    both = ane(input_ids, qk_mask=qk_mask, k_mask=k_mask, output_mask=output_mask)

assert k_only.shape == qk_only.shape
assert k_only.shape == both.shape

print("output shape:", both.shape)

print("k v. qk psnr:", compute_psnr(k_only, qk_only))
print("k v. both psnr:", compute_psnr(k_only, both))
print("qk v. both psnr:", compute_psnr(qk_only, both))


