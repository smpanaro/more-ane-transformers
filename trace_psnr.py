import torch
from ane_gpt2 import GPT as ANEGPT
import numpy as np
import coremltools as ct

"""
Compare the PSNR for a saved mlpackage with a model
and its traced version.
"""

def compute_psnr(a, b):
    """ Compute Peak-Signal-to-Noise-Ratio across two numpy.ndarray objects
    """
    max_b = np.abs(b).max()
    sumdeltasq = 0.0

    sumdeltasq = ((a - b) * (a - b)).sum()

    sumdeltasq /= b.size
    sumdeltasq = np.sqrt(sumdeltasq)

    eps = 1e-5
    eps2 = 1e-10
    psnr = 20 * np.log10((max_b + eps) / (sumdeltasq + eps2))

    return psnr

token_predictor = ANEGPT.from_pretrained("gpt2").eval()

random_tokens = torch.randint(10000, (1,10,))
inputs_dict = token_predictor.build_inputs(random_tokens, pad_to_length=512, pad_token_id=350)
print(token_predictor(inputs_dict["input_ids"], inputs_dict["qk_mask"]).shape)
print(f"Tracing the model with {inputs_dict['input_ids']}")
traced_token_predictor = torch.jit.trace(token_predictor, (inputs_dict["input_ids"], inputs_dict["qk_mask"], inputs_dict["k_mask"]))

print("Traced, diffing...")
random_tokens = torch.randint(10000, (1,10,))
inputs_dict = ANEGPT.build_inputs(random_tokens, pad_to_length=512, pad_token_id=350)

with torch.no_grad():
    og_out = token_predictor(inputs_dict["input_ids"], inputs_dict["qk_mask"], inputs_dict["k_mask"])
    tr_out = traced_token_predictor(inputs_dict["input_ids"], inputs_dict["qk_mask"], inputs_dict["k_mask"])

assert og_out.shape == tr_out.shape

psnr = compute_psnr(og_out.numpy(), tr_out.numpy())
print("psnr:", psnr)

print("Comparing with mlpackage")
model = ct.models.model.MLModel("gpt2_2023_03_24-08_02_53_PM.mlpackage",
                                compute_units=ct.ComputeUnit.ALL)

cm_out = model.predict(inputs_dict)["logits"]
assert tr_out.shape == cm_out.shape

print("coreml-traced psnr:", compute_psnr(tr_out.numpy(), cm_out))
print("coreml-og     psnr:", compute_psnr(og_out.numpy(), cm_out))