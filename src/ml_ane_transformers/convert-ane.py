import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import coremltools as ct
import numpy as np
from datetime import datetime
from .ane_gpt2 import GPT as ANEGPT
from src.utils.psnr import compute_psnr

"""
Convert a ANE-optimized nanoGPT to CoreML.
"""

file_suffix = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

model_name = "gpt2"
# model_name = "ckiplab/gpt2-tiny-chinese"
model_filename = model_name.split("/")[-1] + "_" + file_suffix

retrace = True
if retrace:
    print(f"Loading model {model_name}")
    # token_predictor = AutoModelForCausalLM.from_pretrained(model_name, torchscript=True).eval()
    token_predictor = ANEGPT.from_pretrained(model_name).eval()

    random_tokens = torch.randint(10000, (1,10,))
    inputs_dict = token_predictor.build_inputs(random_tokens, pad_to_length=512, pad_token_id=350)
    print(f"Tracing the model with {inputs_dict['input_ids']}")
    input_ids, qk_mask, k_mask, output_mask = [inputs_dict[k] for k in\
                                            ["input_ids", "qk_mask", "k_mask", "output_mask"]]
    del inputs_dict["k_mask"]
    # Exclude k_mask. It's a no-op for next-token prediction.
    traced_token_predictor = torch.jit.trace(token_predictor, (input_ids, output_mask))

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
    # TODO: This may only be a problem for "gpt", never tried the larger variants.
    # Edit: Much later, it's very reasonable that the mean loses precision in fp16.
    # This should be better in Sonoma.
    return op.op_type not in ["reduce_mean"] or "channels_mean" not in op.name

compute_precision=ct.precision.FLOAT16
# compute_precision=ct.transform.FP16ComputePrecision(op_selector)
mlmodel = ct.convert(
    traced_token_predictor,
    inputs=[
        ct.TensorType(name="input_ids", shape=[1, 512], dtype=np.int32),
        # ct.TensorType(name="qk_mask", shape=[1, 512, 1, 512], dtype=np.float32),
        ct.TensorType(name="output_mask", shape=[1], dtype=np.int32),
    ],
    outputs=[
        ct.TensorType(name="logits", dtype=np.float32),
    ],
    compute_precision=compute_precision,
    # minimum_deployment_target=ct.target.macOS13,
    convert_to="mlprogram",
)


print("Conversion finished")
suffix = ""
if compute_precision == ct.precision.FLOAT32:
    suffix="-f32"

# Save first, sometimes CoreML segfaults.
print("Saving")
mlmodel.save(f"{model_filename}{suffix}-trash-nosplit-allf16.mlpackage")

# Always compare in float32 so we don't overflow.
with torch.no_grad():
    og_out = token_predictor(input_ids, output_mask=output_mask).to(torch.float32)
    tr_out = traced_token_predictor(input_ids, output_mask=output_mask).to(torch.float32)
print({k: f"{v.shape}-{v.dtype}" for k,v in inputs_dict.items()})
del inputs_dict["qk_mask"]
# del inputs_dict["k_mask"]
cm_out = mlmodel.predict(inputs_dict)
cm_out = torch.from_numpy(cm_out["logits"]).to(torch.float32)

assert og_out.shape == cm_out.shape, f"{og_out.shape} != {cm_out.shape}"
assert og_out.dtype == cm_out.dtype, f"{og_out.dtype} != {cm_out.dtype}"


print("this should be quite high. probably >200 or more.")
print("traced-original psnr:", compute_psnr(tr_out.numpy(), og_out.numpy()))
print("\nthese should be >60, ideally much higher.") # otherwise you're only going to generate gibberish
print("coreml-traced   psnr:", compute_psnr(cm_out.numpy(), tr_out.numpy()))
print("coreml-original psnr:", compute_psnr(cm_out.numpy(), og_out.numpy()))
# np.testing.assert_allclose(og_out.numpy(), tr_out.numpy(), atol=1e-5, rtol=1e-4)
# np.testing.assert_allclose(cm_out, tr_out.numpy(), atol=1e-5, rtol=1e-4)
