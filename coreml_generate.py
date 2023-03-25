from ane_gpt2 import GPT as AneGPT
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import coremltools as ct
from stopwatch import Stopwatch

print("Loading tokenizer...")
model_name = "gpt2"
tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token_id = tok.eos_token_id
print("Loaded tokenizer.")


print("Loading model...")
model = ct.models.model.MLModel("gpt2_2023_03_25-12_35_10_AM-conv2dfp16.mlpackage",
                                compute_units=ct.ComputeUnit.ALL)
print("Loaded ANE model.")
print(model)

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

text = "What is the answer to life, the universe, and everything?"
inputs = tok(text, return_tensors="pt")
print("Tokenized initial inputs:", inputs["input_ids"].shape)
ane_inputs = AneGPT.build_inputs(inputs['input_ids'], pad_to_length=512, pad_token_id=tok.pad_token_id)
print("Generated initial inputs:")
print({k: v.shape for k,v in ane_inputs.items()})
print({k: v.dtype for k,v in ane_inputs.items()})
# print({k: v.__class__ for k,v in ane_inputs.items()})

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

NUM_INFERENCES = 20

relevant_tokens = without_pad(ane_inputs["input_ids"])
for i in range(NUM_INFERENCES):
    next_index = len(relevant_tokens[0]) - 1
    ane_inputs = AneGPT.build_inputs(relevant_tokens, pad_to_length=512, pad_token_id=tok.pad_token_id)

    stopwatch.start()
    # Hanging here? It's very likely your intputs are the wrong shape and/or types.
    logits = model.predict(ane_inputs)["logits"]
    stopwatch.stop()

    ane_next = sample(logits[:, [next_index], :]) #ane_inputs['input_ids'], qk_mask=ane_inputs['qk_mask']))

    # Helpful for debugging nonsense generations.
    # print(torch.topk(torch.from_numpy(logits), 20, dim=-1).indices[:, :20, :])
    # print("chose", ane_next, "from idx:", next_index)

    relevant_tokens = torch.cat((relevant_tokens.squeeze(), torch.tensor([ane_next]))).unsqueeze(0)
    print("->", tok.decode(relevant_tokens.squeeze()))

per_inference = "{:.{}f}ms".format((stopwatch.duration / NUM_INFERENCES) * 1000, 2)
print(model.compute_unit)
print(stopwatch, "total")
print(f"{per_inference}/it")