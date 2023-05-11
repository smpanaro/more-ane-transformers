import numpy as np
from models.pythia import GPT as Pythia
import torch
import plotille

"""
Print out statistics about model weights.
"""

models = [
    # "pythia-70m",
    # "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
]

@torch.no_grad()
def print_weight_stats(model_name):
    print("model", model_name)
    model = Pythia.from_pretrained(model_name).eval()

    # Print the min/max/mean across all params.
    # for name, param in model.named_parameters():
    #     print(name, param.shape, param.dtype, param.min(), param.max(), param.mean())

    all_params = []
    for name, param in model.named_parameters():
        all_params.append(param.flatten()) # a tensor
    all_params = torch.cat(all_params, dim=0)

    print("all params", all_params.shape, all_params.dtype, all_params.min(), all_params.max(), all_params.mean())

    counts, bins = np.histogram(all_params.numpy(), bins=10)
    print(plotille.hist_aggregated(
        counts,
        bins,
        width=80,
        log_scale=True,
        linesep='\n',
        lc=None,
        bg=None,
        color_mode='names',
    ))

    print("------\n")

for m in models:
    print_weight_stats(m)
