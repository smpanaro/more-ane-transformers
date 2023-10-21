import coremltools as ct
import argparse
import torch
import numpy as np
from src.utils.model_proxy import MLModelProxy
from models.gpt2 import GPT as GPT2
from models.pythia import GPT as Pythia
from src.utils.psnr import compute_psnr
import sys

"""
Check PSNR between a CoreML model and a non-CoreML model.
Over 60 means there was little loss in the conversion process.
"""

all_names = GPT2.model_names() + Pythia.model_names()

parser = argparse.ArgumentParser(description='Load a CoreML .mlpackage and check the parity of its outputs with the equivalent PyTorch model.')
parser.add_argument('mlmodelc_path', help='path to .mlmodelc/.mlpackage file', default="gpt2.mlpackage", type=str)
parser.add_argument('model_name', choices=all_names, default="gpt2", type=str)

args = parser.parse_args()

model_class = GPT2 if "gpt2" in args.model_name else Pythia
baseline_model = model_class.from_pretrained(args.model_name)

# Not all models work on CPU_ONLY (e.g. pythia-70m)
mlmodel = MLModelProxy(args.mlmodelc_path, ct.ComputeUnit.CPU_AND_NE)

def jaccard(x,y):
    z=set(x).intersection(set(y))
    a=float(len(z))/(len(x)+len(y)-len(z))
    return a

psnrs = []
sample_inputs = baseline_model.sample_inputs()
all_mlmodel_outputs = {} if "kv_cache" not in sample_inputs else {
    "new_kv_cache": sample_inputs["kv_cache"],
}

def make_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x

for i in range(8):
    input_ids = torch.randint(10_000, (1,128,))
    inputs = model_class.build_inputs(input_ids, all_mlmodel_outputs, pad_to_length=512, pad_token_id=0)
    torch_inputs = {k: make_torch(v) for k,v in inputs.items()}

    with torch.no_grad():
        baseline_out = baseline_model(**torch_inputs)
        baseline_out = baseline_out[0] if isinstance(baseline_out, tuple) else baseline_out
        baseline_out = baseline_out.to(torch.float32)
    # Hanging here? It's very likely your intputs are the wrong shape and/or types.
    print("predicting with mlmodel")#, input_ids.shape, input_ids.dtype)
    all_mlmodel_outputs = mlmodel.predict(inputs, {})
    mlmodel_out = torch.from_numpy(all_mlmodel_outputs["logits"]).to(torch.float32)

    assert baseline_out.shape == mlmodel_out.shape, f"{baseline_out.shape} != {mlmodel_out.shape}"
    assert baseline_out.dtype == mlmodel_out.dtype, f"{baseline_out.dtype} != {mlmodel_out.dtype}"

    psnr = compute_psnr(baseline_out, mlmodel_out)
    print("PSNR:", psnr)
    psnrs.append(psnr)

    for k in [80]: #range(40, 400, 40):
        print("k:", k)
        baseline_topk = torch.topk(torch.nn.functional.softmax(baseline_out, dim=-1), k)
        coreml_topk = torch.topk(torch.nn.functional.softmax(mlmodel_out, dim=-1), k)
        # print("baseline_topk:", baseline_topk.indices)
        # print("coreml_topk:", coreml_topk.indices)

        topk_psnr = compute_psnr(baseline_out[:, :, baseline_topk.indices], mlmodel_out[:, :, baseline_topk.indices])
        print("topk PSNR:", topk_psnr)
        # closer to 1 is better
        print("jaccard topk", jaccard(baseline_topk.indices.flatten().tolist(), coreml_topk.indices.flatten().tolist()))

        kl_div = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(mlmodel_out, dim=-1), torch.nn.functional.softmax(baseline_out, dim=-1), reduction="batchmean")
        # clsoer to 0 is better
        print("kl div", kl_div.item())
    print("")

print("Mean PSNR:", np.average(psnrs))
print("Median PSNR:", np.median(psnrs))