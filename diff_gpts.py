from ane_gpt2 import GPT as AneGPT
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

model_name = "gpt2"
hf_model = AutoModelForCausalLM.from_pretrained(model_name)
hf_model.eval()
print("Loaded HF model.")
ane_model = AneGPT.from_pretrained(model_name)
ane_model.eval()
print("Loaded ANE model.")

tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token_id = tok.eos_token_id

def sample(logits, temperature=0.8, top_k=200):
    print("logits.shape", logits.shape)
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
hf_inputs = {k: v.clone() for k,v in inputs.items()}
ane_inputs = ane_model.build_inputs(inputs['input_ids'].clone())
# print("og shape", inputs["input_ids"].shape)
for _ in range(20):
    with torch.no_grad():
        # print('input.shape', ane_inputs['input_ids'].shape)
        ane_inputs = ane_model.build_inputs(ane_inputs['input_ids'])
        ane_next = sample(ane_model(ane_inputs['input_ids'], qk_mask=ane_inputs['qk_mask']))
        ane_inputs['input_ids'] = torch.cat((ane_inputs['input_ids'], ane_next.unsqueeze(-1)), dim=-1)
        print("ANE: ", tok.decode(ane_inputs["input_ids"].squeeze()))


        hf_next = sample(hf_model(**hf_inputs)['logits'])

        hf_inputs["input_ids"] = torch.cat((hf_inputs["input_ids"], hf_next.unsqueeze(-1)), dim=-1)
        hf_inputs["attention_mask"] = torch.cat((hf_inputs["attention_mask"], torch.tensor([[1]])), dim=-1)

        print("HF:", tok.decode(hf_inputs["input_ids"].squeeze()))


# For a less subjective analysis, see the layerwise_comparison.py
