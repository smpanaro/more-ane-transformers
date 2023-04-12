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

"""
Convert a slightly modified nanoGPT or Huggingface pythia to CoreML.
"""

all_names = GPT2.model_names() + Pythia.model_names()

parser = argparse.ArgumentParser(description='Convert a model to CoreML.')
parser.add_argument('--model_name', choices=all_names, default="gpt2", type=str)
parser.add_argument('--low_memory', help="use less memory at the cost of being slower. useful for large models.", action="store_true")
args = parser.parse_args()

file_suffix = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

model_name = args.model_name
model_filename = model_name.split("/")[-1] + "_" + file_suffix

if model_name in ['gpt2-large', 'gpt2-xl']:
    print("WARNING: This model is large and will be slow/impossible to convert without "+
           "installing a modified version of coremltools. Consider using the pre-converted ones instead.")

retrace = True
if retrace:
    print(f"Loading model {model_name}...")
    model_class = GPT2 if model_name.startswith("gpt2") else Pythia
    token_predictor = model_class.from_pretrained(model_name).eval()

    input_ids = torch.randint(10000, (1,512,))
    output_mask = torch.randint(512, (1,))
    print(f"Tracing the model with {input_ids.shape}...")
    with silence_known_trace_warnings(model_name):
        traced_token_predictor = torch.jit.trace(token_predictor, (input_ids, output_mask))
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
if model_name in ["gpt2"]:
    print("Using float32 for layer_norm otherwise the precision lost is too large.")
    print("Larger models can use all float16.") #... and run purely on the neural engine.
    compute_precision=ct.transform.FP16ComputePrecision(op_selector)

if args.low_memory:
    del token_predictor
    gc.collect()

mlmodel = ct.convert(
    traced_token_predictor,
    inputs=[
        ct.TensorType(name="input_ids", shape=[1, 512], dtype=np.int32),
        ct.TensorType(name="output_mask", shape=[1], dtype=np.int32),
    ],
    outputs=[
        ct.TensorType(name="logits", dtype=np.float32),
    ],
    compute_precision=compute_precision,
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

pretty_name = {
    "gpt2": "gpt2 (124M)",
    "gpt2-medium": "gpt2-medium (350M)",
    "gpt2-large": "gpt2-large (774M)",
    "gpt2-xl": "gpt2-xl (1558M)",
}.get(model_name, model_name)
mlmodel.short_description = f"{pretty_name} for text generation. Based on nanoGPT. Optimized for Apple Neural Engine."
mlmodel.input_description["input_ids"] = "Input tokens. e.g. from the huggingface gpt2 tokenizer. Pad to the full length with 50256 (eos)."
mlmodel.input_description["output_mask"] = "A single element array with the index of your sequence to predict. If your non-padded input length was N, pass [N-1]."
mlmodel.output_description["logits"] = "Predictions for the element of input_ids specified by output_mask in the shape (1, 1, 50257). "
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
    og_out = token_predictor(input_ids, output_mask).to(torch.float32)
    tr_out = traced_token_predictor(input_ids, output_mask).to(torch.float32)
input_ids = input_ids.int()
output_mask = output_mask.int()
# Hanging here? It's very likely your intputs are the wrong shape and/or types.
print("predicting with mlmodel")#, input_ids.shape, input_ids.dtype)
cm_out = mlmodel.predict({"input_ids": input_ids.numpy(), "output_mask": output_mask.numpy()})
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
    print("1. Install coremltools from source if you have <= version 6.2") # Sorry.
    print(f"2. Run: python -m src.experiments.chunk_model --mlpackage-path {model_filename}.mlpackage -o .")
    print("3. Edit src/experiments/make_pipeline.py to use the chunked files written by the above command.")
    print("4. Run: python -m src.experiments.make_pipeline")
    print("Use the output *-pipeline.mlpackage with generate.py as usual.")