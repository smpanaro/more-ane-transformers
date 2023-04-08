from src.ml_ane_transformers.ane_gpt2 import GPT as AneGPT
from src.utils.model_proxy import MLModelProxy
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import coremltools as ct
from stopwatch import Stopwatch
from models.gpt2 import GPT as NanoGPT
import argparse
import sys
import os
import glob
from collections import OrderedDict
import subprocess

"""
Load a CoreML model and use it to generate text.
"""

os.environ["TOKENIZERS_PARALLELISM"] = "true"

compute_unit_by_name = OrderedDict([
    ("All", ct.ComputeUnit.ALL),
    ("CPUOnly", ct.ComputeUnit.CPU_ONLY),
    ("CPUAndGPU", ct.ComputeUnit.CPU_AND_GPU),
    ("CPUAndANE", ct.ComputeUnit.CPU_AND_NE),
])

parser = argparse.ArgumentParser(description='Load a CoreML modelpackage and generate some text.')
parser.add_argument('--model_path', help='path to .mlpackage file', default="gpt2.mlpackage", type=str)
parser.add_argument('--input_prompt', help='input prompt for the model', default="Before boarding your rocket to Mars, remember to pack these items:", type=str)
parser.add_argument('--compute_unit', help='compute unit', type=str, choices=list(compute_unit_by_name.keys()), default="All")
parser.add_argument('--length', help='number of new tokens to generate', type=int, default=40)
parser.add_argument('--verbose', help='print verbose logs', type=bool, default=False)
parser.add_argument('--wait', help='wait for confirmation before loading the model (ie to attach a debugger)', action="store_true")
parser.add_argument('--use-mlpackage', help='don\'t automatically generate a mlmodelc and use it. dramatically slower but useful for debugging this script.', action="store_true")

args = parser.parse_args()

if not args.model_path.endswith('.mlpackage') and not args.model_path.endswith('.mlmodelc') :
    print('Error: Model path must end in .mlpackage (or .mlmodelc if you know what you\'re doing)')
    sys.exit(1)

# Special handling for first-time run.
if not os.path.exists(args.model_path) and args.model_path == "gpt2.mlpackage":
    files = glob.glob('gpt2*.mlpackage')
    files = sorted(files, key=lambda x: os.path.getmtime(x))
    if len(files) == 0:
        print(f"Couldn't find {args.model_path}. Either use the --model_path argument or run convert.py to generate one.")
        sys.exit(1)
    args.model_path = files[-1]

compute_unit = compute_unit_by_name[args.compute_unit]

def vprint(*pargs, **kwargs):
    if args.verbose:
        print(*pargs, **kwargs)

vprint("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained("gpt2" if args.model_path.startswith("gpt2") else "EleutherAI/pythia-6.9b")
tok.pad_token_id = tok.eos_token_id
vprint("Loaded tokenizer.")

if args.wait:
    input("Press Enter to continue.")

# Compile to make generations 2-n much much faster.
base_path = args.model_path.replace(".mlpackage/", "").replace(".mlmodelc/", "").replace(".mlpackage", "").replace(".mlmodelc", "")
mlpackage_path = base_path + ".mlpackage"
mlmodelc_path = base_path + ".mlmodelc"
has_compiled_model = os.path.exists(mlmodelc_path)
if not has_compiled_model:
    # Looking to turn this off? As far as I know it's not worth it.
    # Generating text from a mlpackage does this same compilation every time (slow) and
    # it doesn't get cached so you will actually use _more_ disk space without this.
    # It's also much faster to load the model this way. For the xl model this will
    # take model loading from 1.5 minutes to 2.5 seconds.
    print("Compiling model. This first run will be slow but all subsequent runs will be significantly faster.")
    cmd = f"xcrun coremlcompiler compile {mlpackage_path} ."
    compile_result = subprocess.run(cmd, shell=True)
    has_compiled_model = compile_result.returncode == 0
    if not has_compiled_model:
        print("Failed to compile. Please open an issue (https://github.com/smpanaro/more-ane-transformers/issues) and include the following:")
        print(f"code: {compile_result.returncode}\nstdout: {compile_result.stdout}\nstderr: {compile_result.stderr}")
        print("Predicting using the (slow) mlpackage method.")

if has_compiled_model and not os.path.exists(mlpackage_path):
    # TODO: Dump metadata to disk instead so you can keep just the compiled model.
    print(f"No matching mlpackage found for {mlmodelc_path}. Can't predict without that.")
    print(f"It should be at: {mlpackage_path}")
    sys.exit(1)

# nano = NanoGPT.from_pretrained("gpt2").eval()
print(f"Loading model from path {mlmodelc_path if has_compiled_model else mlpackage_path} using {compute_unit}...")
load_stopwatch = Stopwatch(3)
model, model_with_metadata = None, None
if has_compiled_model:
    model = MLModelProxy(mlmodelc_path, compute_unit)
    # So we can inspect and see what the inputs are.
    model_with_metadata = ct.models.model.MLModel(mlpackage_path, compute_units=compute_unit, skip_model_load=True)
else:
    model = ct.models.model.MLModel(mlpackage_path, compute_units=compute_unit)
    model_with_metadata = model
load_stopwatch.stop()
print(f"Loaded model in {load_stopwatch}.")
# print(model)

def sample(logits, temperature=0.8, top_k=20):
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    # pluck the logits at the final step and scale by desired temperature
    logits = logits[:, -1, :] / temperature
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.multinomial(probs.squeeze(), num_samples=1)

text = args.input_prompt
inputs = tok(text, return_tensors="pt")
vprint("Tokenized initial inputs:", inputs["input_ids"].shape)
ane_inputs = AneGPT.build_inputs(inputs['input_ids'], pad_to_length=512, pad_token_id=tok.pad_token_id)
vprint("Generated initial inputs:")
vprint({k: v.shape for k,v in ane_inputs.items()})
vprint({k: v.dtype for k,v in ane_inputs.items()})
# vprint({k: v.__class__ for k,v in ane_inputs.items()})

def get_start_idx(ids):
    ids = ids.tolist()[0]
    if tok.pad_token_id in ids:
        return ids.index(tok.pad_token_id)
    return len(ids)

def from_numpy(d):
    return {k: torch.from_numpy(v) for k,v in d.items()}

def without_pad(ids):
    return ids[ids != tok.pad_token_id].unsqueeze(0)

stopwatch = Stopwatch(3)
stopwatch.stop()
stopwatch.reset()

NUM_INFERENCES = args.length

input_keys = set([f.name for f in model_with_metadata.input_description._fd_spec])

relevant_tokens = without_pad(ane_inputs["input_ids"])
for i in range(NUM_INFERENCES):
    next_index = len(relevant_tokens[0]) - 1
    ane_inputs = AneGPT.build_inputs(relevant_tokens, pad_to_length=512, pad_token_id=tok.pad_token_id)
    ane_inputs = {k:v for k,v in ane_inputs.items() if k in input_keys}

    # attention_mask = ane_inputs["k_mask"].squeeze().unsqueeze(0)
    # print(attention_mask.shape)
    stopwatch.start()
    # Hanging here? It's very likely your intputs are the wrong shape and/or types.
    logits = model.predict(ane_inputs)["logits"] # nano
    # logits = nano(ane_inputs["input_ids"], attention_mask)
    stopwatch.stop()

    # If the model does not pre-select the next token logits, do so now.
    if logits.shape[1] > 1:
        logits = logits[:, [next_index], :]

    ane_next = sample(logits) #ane_inputs['input_ids'], qk_mask=ane_inputs['qk_mask']))

    # Helpful for debugging nonsense generations.
    # print(torch.topk(torch.from_numpy(logits), 20, dim=-1).indices[:, :20, :])
    # print("chose", ane_next, "from idx:", next_index)

    relevant_tokens = torch.cat((relevant_tokens.squeeze(), torch.tensor([ane_next]))).unsqueeze(0)
    if i == 0:
        print(f"\n\033[95m[Prompt] {tok.decode(relevant_tokens.squeeze())}\033[0m", end="")
    else:
        print(tok.decode(ane_next), end="")
    sys.stdout.flush()

print("\n\n---stats---")
per_inference = "{:.{}f}ms".format((stopwatch.duration / NUM_INFERENCES) * 1000, 2)
print("Compute Unit:", args.compute_unit)
print(stopwatch, "total")
print(f"{per_inference}/it")