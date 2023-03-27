import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import coremltools as ct
import numpy as np
from datetime import datetime
from baseline_gpt2 import GPT
from psnr import compute_psnr

"""
Convert a slightly modified nanoGPT to CoreML. Originally intended as
a performance baseline (hence the filename) but realized this is faster
than the ANE-optimized model and tuned it from there.
"""

file_suffix = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

model_name = "gpt2"
model_filename = model_name.split("/")[-1] + "_" + file_suffix

retrace = True
if retrace:
    print(f"Loading model {model_name}...")
    token_predictor = GPT.from_pretrained(model_name).eval()

    input_ids = torch.randint(10000, (1,512,))
    print(f"Tracing the model with {input_ids.shape}...")
    traced_token_predictor = torch.jit.trace(token_predictor, (input_ids))
else:
    print("Loading from saved file.")
    traced_token_predictor = torch.jit.load(f"{model_filename}.pt")

# print(traced_token_predictor)

print("Trace finished.")
print("Beginning conversion...")

def op_selector(op):
    """
    Return true to use float16 for the op. Must be f16 to run on Neural Engine.

    You can find op_type by looking in Netron and/or print out the op type/name here
    (usually the names contain a variable name).
    """
    # LayerNorm is where we lose most of our precision. From experiments
    # in omptimizing for ANE, it's most likely the computing the first mean,
    # but using the non-optimized version we have to float32 the whole layer norm.
    return op.op_type not in ["layer_norm"]

compute_precision=ct.precision.FLOAT16
if model_name == "gpt2":
    print("Using float32 for layer_norm otherwise the precision lost is too large.")
    print("Larger models can use all float16.")
    compute_precision=ct.transform.FP16ComputePrecision(op_selector)
mlmodel = ct.convert(
    traced_token_predictor,
    inputs=[
        ct.TensorType(name="input_ids", shape=[1, 512], dtype=np.int32),
    ],
    outputs=[
        ct.TensorType(name="logits", dtype=np.float32),
    ],
    compute_precision=compute_precision,
    convert_to="mlprogram",
)

print("Conversion finished.")

pretty_name = {
    "gpt2": "gpt2 (124M)",
    "gpt2-medium": "gpt2-medium (350M)",
    "gpt2-large": "gpt2-large (774M)",
    "gpt2-xl": "gpt2-xl (1558M)",
}.get(model_name, model_name)
mlmodel.short_description = f"{pretty_name} for text generation. Based on nanoGPT. Optimized for Apple Neural Engine."
mlmodel.input_description["input_ids"] = "Input tokens. e.g. from the huggingface gpt2 tokenizer. Pad to the full length with 50256 (eos)."
mlmodel.output_description["logits"] = "Predictions for every element of input_ids in the shape (1, 512, 50257). If your non-padded input length was N, look at index [0][N-1]."
mlmodel.user_defined_metadata["Converted By"] = "http://twitter.com/flat"

suffix = ""
if compute_precision == ct.precision.FLOAT32:
    suffix="-f32"

print("Saving...")

# Workaround to save metadata: https://github.com/apple/coremltools/issues/1680
to_save = ct.models.MLModel(mlmodel._spec,
                  weights_dir=mlmodel._weights_dir,
                  is_temp_package=True)
to_save.save(f"{model_filename}{suffix}.mlpackage")

# Always compare in float32 so we don't overflow.
with torch.no_grad():
    og_out = token_predictor(input_ids).to(torch.float32)
    tr_out = traced_token_predictor(input_ids).to(torch.float32)
input_ids = input_ids.int()
print("predicting on mlmodel", input_ids.shape, input_ids.dtype)
cm_out = mlmodel.predict({"input_ids": input_ids.numpy()})
cm_out = torch.from_numpy(cm_out["logits"]).to(torch.float32)
print("predicted")

assert og_out.shape == cm_out.shape, f"{og_out.shape} != {cm_out.shape}"
assert og_out.dtype == cm_out.dtype, f"{og_out.dtype} != {cm_out.dtype}"

print("this should be quite high. probably >200 or more.")
print("traced-original psnr:", compute_psnr(og_out.numpy(), tr_out.numpy()))
print("\nthese should be >60, ideally much higher.")
print("coreml-traced   psnr:", compute_psnr(tr_out.numpy(), cm_out.numpy()))
print("coreml-original psnr:", compute_psnr(og_out.numpy(), cm_out.numpy()))
# np.testing.assert_allclose(og_out.numpy(), tr_out.numpy(), atol=1e-5, rtol=1e-4)
# np.testing.assert_allclose(cm_out, tr_out.numpy(), atol=1e-5, rtol=1e-4)
