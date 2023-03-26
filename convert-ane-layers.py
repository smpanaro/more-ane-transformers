import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import coremltools as ct
import numpy as np
from datetime import datetime
from ane_gpt2 import GPT as ANEGPT

"""
Experimental setup for going layer-by-layer to see which
layers cause a drop in PSNR after conversion to CoreML.
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

file_suffix = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

class TestNet(torch.nn.Module):
    """
    Wraps a ANEGPT model so individual layers and partial
    combinations of layers can be tested.
    """
    def __init__(self, ane: ANEGPT):
        super().__init__()
        self.ane = ane

    def forward(self, x, qk_mask=None, k_mask=None):
        device = x.device
        b, t = x.size()
        # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        tok_emb = self.ane.transformer.wte(x) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.ane.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.ane.transformer.drop(tok_emb + pos_emb) # psnr: f32:212, f16:99

        x = x.transpose(1, 2).unsqueeze(2)
        for h in self.ane.transformer.h:
            x = h(x, qk_mask, k_mask)
            # x = h.ln_1(x)
            # x = h.attn(x, qk_mask=qk_mask, k_mask=k_mask) # psnr f16:89
            # x = h.ln_2(x) # psnr f16:77
            # x = h.mlp(x)  # psnr f16:62

        # x = x.permute(3, 1, 0, 2)
        # x = self.ane.lm_head(x).squeeze().unsqueeze(0)
        return x

model_name = "gpt2"
print(f"Loading model {model_name}")
ane = ANEGPT.from_pretrained(model_name).eval()

token_predictor = TestNet(ane).eval()
# token_predictor = ane # test the whole model

random_tokens = torch.randint(30000, (1,10,))
inputs_dict = ANEGPT.build_inputs(random_tokens, pad_to_length=512, pad_token_id=350)
input_ids, qk_mask, k_mask = inputs_dict["input_ids"], inputs_dict["qk_mask"], inputs_dict["k_mask"]

print(f"Tracing the model with {input_ids.shape}")

traced_token_predictor = torch.jit.trace(token_predictor, (input_ids, qk_mask, k_mask))

print(traced_token_predictor)

print("Trace finished")
print("Beginning conversion")

mlmodel = ct.convert(
    traced_token_predictor,
    inputs=[
        ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
        ct.TensorType(name="qk_mask", shape=[1, 512, 1, 512], dtype=np.float32),
        ct.TensorType(name="k_mask", shape=[1, 512, 1, 1], dtype=np.float32),
    ],
    outputs=[
        ct.TensorType(name="logits", dtype=np.float32),
    ],
    compute_precision=ct.precision.FLOAT16,
    # minimum_deployment_target=ct.target.macOS13,
    convert_to="mlprogram",
)

print("Conversion finished")

with torch.no_grad():
    og_out = token_predictor(input_ids, qk_mask, k_mask).to(torch.float32)
    tr_out = traced_token_predictor(input_ids, qk_mask, k_mask).to(torch.float32)
cm_out = mlmodel.predict({"input_ids": input_ids, "qk_mask": qk_mask, "k_mask": k_mask})
cm_out = torch.from_numpy(cm_out["logits"]).to(torch.float32)

assert og_out.shape == cm_out.shape, f"{og_out.shape} != {cm_out.shape}"
assert og_out.dtype == cm_out.dtype, f"{og_out.dtype} != {cm_out.dtype}"

print("traced-og     psnr:", compute_psnr(og_out.numpy(), tr_out.numpy()))
print("coreml-traced psnr:", compute_psnr(tr_out.numpy(), cm_out.numpy()))
print("coreml-og     psnr:", compute_psnr(og_out.numpy(), cm_out.numpy()))
