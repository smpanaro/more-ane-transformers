from src.ml_ane_transformers.ane_gpt2 import GPT as AneGPT
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
from collections import OrderedDict

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
parser.add_argument('--input_prompt', help='input prompt for the model', default="Before boarding your rocket to Mars, remember to pack these items", type=str)
parser.add_argument('--compute_unit', help='compute unit', type=str, choices=list(compute_unit_by_name.keys()), default="All")
parser.add_argument('--length', help='number of new tokens to generate', type=int, default=40)
parser.add_argument('--verbose', help='print verbose logs', type=bool, default=False)

args = parser.parse_args()

if not args.model_path.endswith('.mlpackage'):
    print('Error: Model path must end in .mlpackage')

compute_unit = compute_unit_by_name[args.compute_unit]

def vprint(*pargs, **kwargs):
    if args.verbose:
        print(*pargs, **kwargs)

vprint("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token_id = tok.eos_token_id
vprint("Loaded tokenizer.")

# nano = NanoGPT.from_pretrained("gpt2").eval()
print(f"Loading model from path {args.model_path} using {compute_unit}...")
load_stopwatch = Stopwatch(3)
model = ct.models.model.MLModel(args.model_path, compute_units=compute_unit)
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

relevant_tokens = without_pad(ane_inputs["input_ids"])
for i in range(NUM_INFERENCES):
    next_index = len(relevant_tokens[0]) - 1
    ane_inputs = AneGPT.build_inputs(relevant_tokens, pad_to_length=512, pad_token_id=tok.pad_token_id)

    # attention_mask = ane_inputs["k_mask"].squeeze().unsqueeze(0)
    # print(attention_mask.shape)
    stopwatch.start()
    # Hanging here? It's very likely your intputs are the wrong shape and/or types.
    # logits = model.predict(ane_inputs)["logits"]
    logits = model.predict({"input_ids": ane_inputs["input_ids"]})["logits"] # nano
    # logits = nano(ane_inputs["input_ids"], attention_mask)
    stopwatch.stop()

    ane_next = sample(logits[:, [next_index], :]) #ane_inputs['input_ids'], qk_mask=ane_inputs['qk_mask']))

    # Helpful for debugging nonsense generations.
    # print(torch.topk(torch.from_numpy(logits), 20, dim=-1).indices[:, :20, :])
    # print("chose", ane_next, "from idx:", next_index)

    relevant_tokens = torch.cat((relevant_tokens.squeeze(), torch.tensor([ane_next]))).unsqueeze(0)
    if i == 0:
        print(f"\n\033[95m{tok.decode(relevant_tokens.squeeze())}\033[0m", end="")
    else:
        print(tok.decode(ane_next), end="")
    sys.stdout.flush()

print("\n\n---stats---")
per_inference = "{:.{}f}ms".format((stopwatch.duration / NUM_INFERENCES) * 1000, 2)
print(args.compute_unit)
print(stopwatch, "total")
print(f"{per_inference}/it")