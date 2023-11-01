from src.ml_ane_transformers.ane_gpt2 import GPT as AneGPT
from src.utils.model_proxy import MLModelProxy
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import coremltools as ct
from stopwatch import Stopwatch
from models.gpt2 import GPT as GPT2
from models.pythia import GPT as Pythia
import argparse
import sys
import os
import glob
import math
from collections import OrderedDict
import subprocess
from  os_signpost import Signposter

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
parser.add_argument('--compute_unit', help='compute unit', type=str, choices=list(compute_unit_by_name.keys()), default="CPUAndANE")
parser.add_argument('--length', help='number of new tokens to generate', type=int, default=40)
parser.add_argument('--verbose', help='print verbose logs', type=bool, default=False)
parser.add_argument('--wait', help='wait for confirmation before loading the model (ie to attach a debugger)', action="store_true")
parser.add_argument('--use-mlpackage', help='don\'t automatically generate a mlmodelc and use it. dramatically slower but useful for debugging this script.', action="store_true")
parser.add_argument('--argmax', help='use deterministic argmax instead of multinomial sampling', action="store_true")
parser.add_argument('--timingstats', help='print verbose timing stats', action="store_true")

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

def get_tokenizer_name(model_path):
    names = GPT2.model_names() + Pythia.model_names()
    tokenizer_lookup = {**GPT2.tokenizer_by_name(), **Pythia.tokenizer_by_name()}
    for n in sorted(names, key=len):
        if model_path.startswith(n):
            return tokenizer_lookup[n]
    print(f"No tokenizer found for {model_path}")
    print(f"Model name must start with one of:")
    print(names)
    return None

tokenizer_name = get_tokenizer_name(args.model_path)
if tokenizer_name is None:
    sys.exit(1)

vprint("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(tokenizer_name)
tok.pad_token_id = tok.eos_token_id
vprint("Loaded tokenizer.")

signposter = Signposter("com.smpanaro.more-ane-transformers", Signposter.Category.PointsOfInterest)

if args.wait:
    print(f"Current PID: {os.getpid()}")
    input("Waiting. Press Enter to continue.")

# total time from model load to eval end
total_stopwatch = Stopwatch(3)

# Compile to make generations 2-n much much faster.
base_path = args.model_path.replace(".mlpackage/", "").replace(".mlmodelc/", "").replace(".mlpackage", "").replace(".mlmodelc", "")
mlpackage_path = base_path + ".mlpackage"
mlmodelc_path = base_path + ".mlmodelc"
has_compiled_model = os.path.exists(mlmodelc_path)
did_compile_model = False
if not has_compiled_model:
    end_compile = signposter.begin_interval("Compile Model")
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
    did_compile_model = True
    end_compile()

if has_compiled_model and not os.path.exists(mlpackage_path):
    # TODO: Dump metadata to disk instead so you can keep just the compiled model.
    print(f"No matching mlpackage found for {mlmodelc_path}. Can't predict without that.")
    print(f"It should be at: {mlpackage_path}")
    sys.exit(1)

# nano = NanoGPT.from_pretrained("gpt2").eval()
print(f"Loading model from path {mlmodelc_path if has_compiled_model else mlpackage_path} using {compute_unit}...")
end_load = signposter.begin_interval("Load Model")
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
end_load()
print(f"Loaded model in {load_stopwatch}.")
# print(model)

@torch.no_grad()
def sample_multinomial(logits, temperature=0.85, top_k=80):
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    # pluck the logits at the final step and scale by desired temperature
    logits = logits[:, -1, :] / temperature
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.multinomial(probs.squeeze(), num_samples=1)

@torch.no_grad()
def sample_argmax(logits):
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits).float()
    return torch.argmax(logits[:, -1, :], dim=-1)
sample = sample_argmax if args.argmax else sample_multinomial

text = args.input_prompt
inputs = tok(text, return_tensors="pt")
vprint("Tokenized initial inputs:", inputs["input_ids"].shape)
ane_inputs = AneGPT.build_inputs(inputs['input_ids'], pad_to_length=512, pad_token_id=tok.pad_token_id)
vprint("Generated initial inputs:")
vprint({k: v.shape for k,v in ane_inputs.items()})
vprint({k: v.dtype for k,v in ane_inputs.items()})
# vprint({k: v.__class__ for k,v in ane_inputs.items()})

def from_numpy(d):
    return {k: torch.from_numpy(v) for k,v in d.items()}

def without_pad(ids):
    return ids[ids != tok.pad_token_id].unsqueeze(0)

NUM_INFERENCES = args.length

input_keys = set([f.name for f in model_with_metadata.input_description._fd_spec])

# Different models take different inputs.
input_builder = None
prompt_input_output_mapping = {}
generation_input_output_mapping = {}
if input_keys == set(["input_ids", "output_mask"]):
    input_builder = Pythia
elif input_keys == set(["input_ids", "full_sequence_length", "kv_cache"]):
    input_builder = GPT2
    prompt_input_output_mapping["kv_cache"] = "prompt_kv_cache"
    prompt_input_output_mapping["fake_key"] = "generation_kv_cache" # needed for transition from prompt -> generation. todo: better API for this.
    generation_input_output_mapping["kv_cache"] = "generation_kv_cache"
    generation_input_output_mapping["fake_key"] = "prompt_kv_cache" # avoid converting to python (speed). todo: better API for this.
else:
    print(f"Unsupported model inputs: {input_keys}.")
    sys.exit(1)

relevant_tokens = without_pad(ane_inputs["input_ids"])
outputs = {}
pad_to_length = 512

input_ids_length = {f.name: f for f in model_with_metadata.input_description._fd_spec}["input_ids"].type.multiArrayType.shape[-1]
prompt_chunks = math.ceil(len(relevant_tokens[0]) / input_ids_length)

prompt_stopwatch = Stopwatch(3)
end_prompt_processing = signposter.begin_interval(f"Process Prompt Chunks")
for i in range(prompt_chunks):
    end_predict = signposter.begin_interval(f"Predict Prompt")
    with signposter.use_interval("Build Inputs"):
        input_args = {
            'input_length': input_ids_length,
            'outputs': outputs,
            'pad_to_length': pad_to_length,
            'pad_token_id': tok.pad_token_id,
            'prompt_chunk_idx': i,
        }
        ane_inputs =  input_builder.build_inputs(relevant_tokens, **input_args)
        ane_inputs = {k:v for k,v in ane_inputs.items() if k in input_keys}

    outputs = model.predict(ane_inputs, prompt_input_output_mapping)
    # Just pass outputs into the next iteration to build up the KV cache (if needed).
    end_predict(f"{i}")

# Last prompt chunk generates the first new token.
with signposter.use_interval("Sample"):
    logits = outputs["logits"]
    # If the model does not pre-select the next token logits, do so now.
    if logits.shape[1] > 1:
        logits = logits[:, [next_index], :]
    ane_next = sample(logits) #ane_inputs['input_ids'], qk_mask=ane_inputs['qk_mask']))
print(f"\n\033[95m[Prompt] {tok.decode(relevant_tokens.squeeze())}\033[0m", end="")
relevant_tokens = torch.cat((relevant_tokens.squeeze(), torch.tensor([ane_next]))).unsqueeze(0)
print(tok.decode(ane_next), end="")
sys.stdout.flush()

end_prompt_processing(f"{prompt_chunks}")
prompt_stopwatch.stop()

# non-prompt generations
generation_stopwatch = Stopwatch(3)
for i in range(NUM_INFERENCES):
    end_predict = signposter.begin_interval(f"Predict Token")
    next_index = len(relevant_tokens[0]) - 1

    with signposter.use_interval("Build Inputs"):
        input_args = {
            'input_length': input_ids_length,
            'outputs': outputs,
            'pad_to_length': pad_to_length,
            'pad_token_id': tok.pad_token_id,
        }
        ane_inputs = input_builder.build_inputs(relevant_tokens, **input_args)
        ane_inputs = {k:v for k,v in ane_inputs.items() if k in input_keys}

    # attention_mask = ane_inputs["k_mask"].squeeze().unsqueeze(0)
    # print(attention_mask.shape)
    # Hanging here? It's very likely your intputs are the wrong shape and/or types.
    outputs = model.predict(ane_inputs, generation_input_output_mapping)
    logits = outputs["logits"] # nano
    # logits = nano(ane_inputs["input_ids"], attention_mask)

    with signposter.use_interval("Sample"):
        # If the model does not pre-select the next token logits, do so now.
        if logits.shape[1] > 1:
            logits = logits[:, [next_index], :]
        ane_next = sample(logits) #ane_inputs['input_ids'], qk_mask=ane_inputs['qk_mask']))

    # Helpful for debugging nonsense generations.
    # print(torch.topk(torch.from_numpy(logits), 20, dim=-1).indices[:, :20, :])
    # print("chose", ane_next, "from idx:", next_index)

    relevant_tokens = torch.cat((relevant_tokens.squeeze(), torch.tensor([ane_next]))).unsqueeze(0)
    print(tok.decode(ane_next), end="")
    sys.stdout.flush()
    end_predict(f"{i}")

generation_stopwatch.stop()
total_stopwatch.stop()

load_duration = "{:.{}f} ms".format(load_stopwatch.duration*1000, 2)
total_duration = "{:.{}f} ms".format(total_stopwatch.duration*1000, 2)
total_prompt_duration = "{:.{}f} ms".format(prompt_stopwatch.duration*1000, 2)
prompt_per_token = "{:.{}f} ms".format((prompt_stopwatch.duration / (prompt_chunks*input_ids_length)) * 1000, 2)
prompt_per_second = "{:.{}f} tokens/s".format((prompt_chunks*input_ids_length) / prompt_stopwatch.duration, 2)
total_generation_duration = "{:.{}f} ms".format(generation_stopwatch.duration*1000, 2)
generation_per_token = "{:.{}f} ms".format((generation_stopwatch.duration / NUM_INFERENCES) * 1000, 2)
generation_per_second = "{:.{}f} tokens/s".format(NUM_INFERENCES / generation_stopwatch.duration, 2)

first_load_caveat = " [uncached load, cached loads will be faster]" if did_compile_model else ""

if args.timingstats:
    kl = 20
    vl = max([len(v) for v in [load_duration, total_prompt_duration, total_generation_duration, total_duration]])
    print(f"\n\n{'---stats---':>{kl}}")
    print(f"{'compute unit:':>{kl}} {args.compute_unit}")
    print(f"{'model load time:':>{kl}} {load_duration:<{vl}}{first_load_caveat}")
    print(f"{'prompt eval time:':>{kl}} {total_prompt_duration:<{vl}} : {prompt_per_token}/token : {prompt_per_second}")
    print(f"{'new token eval time:':>{kl}} {total_generation_duration:<{vl}} : {generation_per_token}/token : {generation_per_second}")
    print(f"{'total time:':>{kl}} {total_duration:<{vl}}")
else:
    print("\n\n---stats---")
    # default to easier to understand stats
    print("Compute Unit:", args.compute_unit)
    print(f"{total_duration} total")
    print(f"{generation_per_token}/token")

if not model.supports_input_output_cache and "kv_cache" in input_keys:
    print("\n🏎️  For a 2-5x speedup, follow the steps in SETUP.md.")
