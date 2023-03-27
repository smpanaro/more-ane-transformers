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

model_name = "gpt2-large"
model_filename = model_name.split("/")[-1] + "_" + file_suffix

retrace = True
if retrace:
    print(f"Loading model {model_name}")
    token_predictor = GPT.from_pretrained(model_name).eval()

    input_ids = torch.randint(10000, (1,512,))
    print(f"Tracing the model with {input_ids.shape}")
    traced_token_predictor = torch.jit.trace(token_predictor, (input_ids))

    #traced_token_predictor.save(f"{model_filename}.pt")
else:
    print("Loading from saved file")
    traced_token_predictor = torch.jit.load(f"{model_filename}.pt")

print(traced_token_predictor)

print("Trace finished")
print("Beginning conversion")

# Going totally F16 is too much. The PSNR drops dramatically and generation is completely garbled.
def op_selector(op):
    """
    Return true to use float16 for the op. Must be f16 to run on Neural Engine.

    You can find op_type by looking in Netron and print out the op names here
    (usually they contain a variable name).
    """
    # All the ops involved in LayerNorm. Keep this in f32.
    # LayerNorm is where we lose most of our precision. Interestingly, it seems
    # the first mean contributes almost all the error.
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


print("Conversion finished")

suffix = ""
if compute_precision == ct.precision.FLOAT32:
    suffix="-f32"

print("Saving")
mlmodel.save(f"baseline-{model_filename}{suffix}-fmixed.mlpackage")

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
