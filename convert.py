import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import coremltools as ct
import numpy as np
from datetime import datetime
from models.gpt2 import GPT as GPT2
from models.pythia import GPT as Pythia
from src.utils.psnr import compute_psnr
from src.utils.trace_warnings import silence_known_trace_warnings
import argparse
import gc
import sys
import platform

"""
Convert a slightly modified nanoGPT or Huggingface pythia to CoreML.
"""

all_names = GPT2.model_names() + Pythia.model_names()

parser = argparse.ArgumentParser(description='Convert a model to CoreML.')
parser.add_argument('--model_name', choices=all_names, default="gpt2", type=str)
parser.add_argument('--low_memory', help="use less memory at the cost of slower conversion. useful for large models.", action="store_true")
parser.add_argument('--float16_mode', choices=['auto', 'force'], default="auto", type=str, help="whether the converted model uses float16 or float32 inputs. 'auto' chooses the fastest that the current device's OS supports")
args = parser.parse_args()

# float16 inference is only supported on macOS13/iOS16 and higher.
supports_float16 = int(platform.mac_ver()[0].split('.')[0]) >= 13
use_float16 = supports_float16 or args.float16_mode == "force"
if not supports_float16:
    print("float16 inputs and outputs are only supported on macOS13/iOS16 and higher.")
    print("Converting with float32 inputs and outputs instead, so you can run it on this device.")
    print("If you plan to deploy to newer device, pass --float16_mode force to use float16 instead.")
if args.float16_mode == "force":
    print("Forcing conversion to use float16 inputs and outputs.")

file_suffix = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

model_name = args.model_name
model_filename = model_name.split("/")[-1] + "_" + file_suffix

retrace = True
if retrace:
    print(f"Loading model {model_name}...")
    model_class = GPT2 if model_filename.startswith("gpt2") else Pythia
    torch_model = model_class.from_pretrained(model_name).eval()

    # input_ids = torch.randint(10000, (1,512,))
    # output_mask = torch.randint(512, (1,))
    # trace_inputs = list(torch_model.sample_inputs().values())

    sample_inputs = torch_model.sample_inputs()
    output_types = torch_model.output_types()
    if not use_float16:
        sample_inputs = {k: v.to(torch.float32) for k,v in sample_inputs.items()}
        output_types = {k: torch.float32 if v == torch.float16 else v for k,v in output_types.items()}

    torch_sample_inputs = list(sample_inputs.values())
    is_multi_output = len(torch_model.output_types()) > 1

    print(f"Tracing the model with {len(torch_sample_inputs)} inputs...")
    with silence_known_trace_warnings(model_name):
        traced_model = torch.jit.trace(torch_model, torch_sample_inputs)
else:
    print("Loading from saved file.")
    traced_model = torch.jit.load(f"{model_filename}.pt")

# print(traced_model)

print("Trace finished.")
print("Beginning conversion...")

def op_selector(op):
    """
    Return true to use float16 for the op. Must be f16 to run on Neural Engine.

    You can find op_type by looking in Netron and/or print out the op type/name here
    (usually the names contain a variable name).
    """
    # LayerNorm is where we lose most of our precision. From experiments
    # in optimizing for ANE, it's most likely the computing the first mean,
    # but using the non-ANE-optimized architecture we have to float32 the whole layer norm.
    # TODO: This may no longer be necessary on iOS17+, try it.
    return op.op_type not in ["layer_norm"]

compute_precision=ct.precision.FLOAT16
if model_name in ["gpt2"]:
    print("Using float32 computation for layer_norm otherwise the precision lost is too large.")
    print("Larger models can use all float16.") #... and run purely on the neural engine.
    compute_precision=ct.transform.FP16ComputePrecision(op_selector)

if args.low_memory:
    del token_predictor
    gc.collect()

coreml_sample_inputs = {k: v.numpy() for k,v in sample_inputs.items()}
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name=k, shape=v.shape, dtype=v.dtype, default_value=np.zeros(v.shape, dtype=v.dtype) if k == "kv_cache" else None)
        for k,v in coreml_sample_inputs.items()
    ],
    outputs=[
        # No better way to convert dtypes?
        ct.TensorType(name=k, dtype=torch.tensor(0, dtype=v).numpy().dtype)
        for k,v in output_types.items()
    ],
    compute_precision=compute_precision,
    minimum_deployment_target=ct.target.iOS16, # To allow float16 inputs + outputs. # TODO: Make optional.
    convert_to="mlprogram",
)

print("Conversion finished.")

if args.low_memory:
    del traced_token_predictor
    gc.collect()

    print("Saving...")
    mlmodel.save(f"{model_filename}.mlpackage")

    del mlmodel
    gc.collect()

    print("Adding metadata...")
    mlmodel = ct.models.MLModel(f"{model_filename}.mlpackage", skip_model_load=True)

# TODO: Clean up.
pretty_name = {
    "gpt2": "gpt2 (124M)",
    "gpt2-medium": "gpt2-medium (350M)",
    "gpt2-large": "gpt2-large (774M)",
    "gpt2-xl": "gpt2-xl (1558M)",
}.get(model_name, model_name)
model_family = [x for x in ["gpt2", "pythia"] if x in model_name][0]
eos_token_id = {"gpt2": 50256, "pythia": 0}[model_family]
based_on = {"gpt2": "nanoGPT", "pythia": "the HuggingFace implementation"}[model_family]
vocab_size = {"gpt2": 50257, "pythia": 50304}[model_family] if model_name != "pythia-6.9b" else 50432

mlmodel.short_description = f"{pretty_name} for text generation. Based on {based_on}. Optimized for Apple Neural Engine."

input_keys = list(sample_inputs.keys())
input_pad_side = {"gpt2": "left", "pythia": "right"}[model_family]
has_output_mask = "output_mask" in input_keys
logits_element_description = "element of input_ids specified by output_mask" if has_output_mask else "next element after input_ids"
input_output_descriptions = {
    # Common
    "input_ids": f"Input tokens. e.g. from the huggingface {model_family} tokenizer. Pad to the full length with {eos_token_id} (eos) on the {input_pad_side}.",
    "logits": f"Predictions for the {logits_element_description} in the shape (1, 1, {vocab_size}). ",

    # KV Cache
    "pos_offset": "The index of the first non-pad token in input_ids.",
    "kv_cache": "Intermediary outputs from the prior prediction. For the first prediction, pass an array of all zeros. For subsequent predictions, pass the new_kv_cache output from the previous prediction.",
    "new_kv_cache": "Intermediary outputs for the next prediction. Pass as the kv_cache input to the next prediction.",
    "qk_mask": "An array that is added to the result of each Q@K matrix multiplication. Pass 0 for values that should be attended to and -1e4 for values that should be ignored. This should be a right triangle full of zeros with the hypotenuse on the top right and the lower-right corner at the bottom right of the matrix. Each leg should be the same length as the un-padded number of input tokens.",

    # No KV Cache
    "output_mask": "A single element array with the index of your sequence to predict. If your non-padded input length was N, pass [N-1].",
}
for k in input_keys:
    mlmodel.input_description[k] = input_output_descriptions[k]
for k in output_types.keys():
    mlmodel.output_description[k] = input_output_descriptions[k]

mlmodel.user_defined_metadata["Converted By"] = "http://twitter.com/flat"
mlmodel.user_defined_metadata["URL"] = "https://github.com/smpanaro/more-ane-transformers"

if not args.low_memory:
    print("Saving...")

# Workaround to save metadata: https://github.com/apple/coremltools/issues/1680
to_save = ct.models.MLModel(mlmodel._spec,
                  weights_dir=mlmodel._weights_dir,
                  is_temp_package=True)
to_save.save(f"{model_filename}.mlpackage")

if args.low_memory:
    print("Skipping model comparison due to low memory mode.")
    print("Conversion complete.")
    sys.exit(0)

# Always compare in float32 so we don't overflow.
with torch.no_grad():
    og_out = torch_model(*torch_sample_inputs)
    og_out = og_out[0] if isinstance(og_out, tuple) else og_out
    og_out = og_out.to(torch.float32)

    tr_out = traced_model(*torch_sample_inputs)
    tr_out = tr_out[0] if isinstance(tr_out, tuple) else tr_out
    tr_out = tr_out.to(torch.float32)

# Hanging here? It's very likely your intputs are the wrong shape and/or types.
print("predicting with mlmodel")#, input_ids.shape, input_ids.dtype)
cm_out = mlmodel.predict(coreml_sample_inputs)
cm_out = torch.from_numpy(cm_out["logits"]).to(torch.float32)

assert og_out.shape == cm_out.shape, f"{og_out.shape} != {cm_out.shape}"
assert og_out.dtype == cm_out.dtype, f"{og_out.dtype} != {cm_out.dtype}"

trace_psnr = compute_psnr(og_out, tr_out)
if trace_psnr < 200:
    print(f"tracing PSNR too low ({trace_psnr}), CoreML model will likely be unusable")

print("\nfinished. these should be >60, ideally much higher (inf is perfect). lower and the model may not be usable")
print("coreml-traced   psnr:", compute_psnr(tr_out.numpy(), cm_out.numpy()))
print("coreml-original psnr:", compute_psnr(og_out.numpy(), cm_out.numpy()))

if model_name in ["gpt2-xl"]:
    print("\n👋 This model is big. It will run fast if you have a recent Mac with a fast GPU.")
    print("If not you can download a version that runs on the Neural Engine from the releases tab on GitHub.")
    print("If you want to build it yourself follow these steps:")
    print("1. Install coremltools >= 6.3")
    print(f"2. Run: python -m src.experiments.chunk_model --mlpackage-path {model_filename}.mlpackage -o .")
    print(f"3. Run: python -m src.experiments.make_pipeline {model_filename}_chunk1.mlpackage")
    print("Use the output *-pipeline.mlpackage with generate.py as usual.")