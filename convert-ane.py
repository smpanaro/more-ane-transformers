import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import coremltools as ct
import numpy as np
from datetime import datetime
from ane_gpt2 import GPT as ANEGPT

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

model_name = "gpt2"
# model_name = "togethercomputer/GPT-JT-6B-v1"
#model_name = "trl-internal-testing/tiny-random-GPTJForCausalLM"
model_filename = model_name.split("/")[-1] + "_" + file_suffix

retrace = True
if retrace:
    print(f"Loading model {model_name}")
    # token_predictor = AutoModelForCausalLM.from_pretrained(model_name, torchscript=True).eval()
    token_predictor = ANEGPT.from_pretrained("gpt2").eval()

    random_tokens = torch.randint(10000, (1,10,))
    inputs_dict = token_predictor.build_inputs(random_tokens, pad_to_length=512, pad_token_id=350)
    print(token_predictor(inputs_dict["input_ids"], inputs_dict["qk_mask"]).shape)
    print(f"Tracing the model with {inputs_dict['input_ids']}")
    traced_token_predictor = torch.jit.trace(token_predictor, (inputs_dict["input_ids"], inputs_dict["qk_mask"], inputs_dict["k_mask"]))

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
    return op.op_type not in ["reduce_mean"] or "channels_mean" not in op.name

# compute_precision=ct.precision.FLOAT16
compute_precision=ct.transform.FP16ComputePrecision(op_selector)
mlmodel = ct.convert(
    traced_token_predictor,
    # Range for the sequence dimension to be between [1, 64]
    inputs=[
        ct.TensorType(name="input_ids", shape=[1, 512], dtype=np.int32),
        ct.TensorType(name="qk_mask", shape=[1, 512, 1, 512], dtype=np.float32),
        ct.TensorType(name="k_mask", shape=[1, 512, 1, 1], dtype=np.float32),
    ],
    outputs=[
        ct.TensorType(name="logits", dtype=np.float32),
    ],
    compute_precision=compute_precision,
    convert_to="mlprogram",
)


print("Conversion finished")

# Always compare in float32 so we don't overflow.
with torch.no_grad():
    og_out = token_predictor(inputs_dict["input_ids"], inputs_dict["qk_mask"], inputs_dict["k_mask"]).to(torch.float32)
    tr_out = traced_token_predictor(inputs_dict["input_ids"], inputs_dict["qk_mask"], inputs_dict["k_mask"]).to(torch.float32)
cm_out = mlmodel.predict(inputs_dict)
cm_out = torch.from_numpy(cm_out["logits"]).to(torch.float32)

assert og_out.shape == cm_out.shape, f"{og_out.shape} != {cm_out.shape}"
assert og_out.dtype == cm_out.dtype, f"{og_out.dtype} != {cm_out.dtype}"


print("this should be quite high. probably >200 or more.")
print("traced-original psnr:", compute_psnr(og_out.numpy(), tr_out.numpy()))
print("\nthese should be >60, ideally much higher.")
print("coreml-traced   psnr:", compute_psnr(tr_out.numpy(), cm_out.numpy()))
print("coreml-original psnr:", compute_psnr(og_out.numpy(), cm_out.numpy()))
# np.testing.assert_allclose(og_out.numpy(), tr_out.numpy(), atol=1e-5, rtol=1e-4)
# np.testing.assert_allclose(cm_out, tr_out.numpy(), atol=1e-5, rtol=1e-4)

suffix = ""
if compute_precision == ct.precision.FLOAT32:
    suffix="-f32"

print("Saving")
mlmodel.save(f"{model_filename}{suffix}.mlpackage")