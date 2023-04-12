import coremltools as ct
import argparse
import torch
import numpy as np
from src.utils.model_proxy import MLModelProxy
from models.gpt2 import GPT as GPT2
from models.pythia import GPT as Pythia
from src.utils.psnr import compute_psnr

"""
Check PSNR between a CoreML model and a non-CoreML model.
Over 60 means there was little loss in the conversion process.
"""

all_names = GPT2.model_names() + Pythia.model_names()

parser = argparse.ArgumentParser(description='Load a CoreML modelpackage and generate some text.')
parser.add_argument('mlmodelc_path', help='path to .mlpackage file', default="gpt2.mlpackage", type=str)
parser.add_argument('model_name', choices=all_names, default="gpt2", type=str)

args = parser.parse_args()


model_class = GPT2 if "gpt2" in args.model_name else Pythia
baseline_model = model_class.from_pretrained(args.model_name)

mlmodel = MLModelProxy(args.mlmodelc_path, ct.ComputeUnit.CPU_AND_NE)

psnrs = []
for i in range(5):
    input_ids = torch.randint(10_000, (1,512,))
    output_mask = torch.randint(512, (1,))
    with torch.no_grad():
        baseline_out = baseline_model(input_ids, output_mask).to(torch.float32)
    input_ids = input_ids.int()
    output_mask = output_mask.int()
    # Hanging here? It's very likely your intputs are the wrong shape and/or types.
    print("predicting with mlmodel")#, input_ids.shape, input_ids.dtype)
    mlmodel_out = mlmodel.predict({"input_ids": input_ids.numpy(), "output_mask": output_mask.numpy()})
    mlmodel_out = torch.from_numpy(mlmodel_out["logits"]).to(torch.float32)

    assert baseline_out.shape == mlmodel_out.shape, f"{baseline_out.shape} != {mlmodel_out.shape}"
    assert baseline_out.dtype == mlmodel_out.dtype, f"{baseline_out.dtype} != {mlmodel_out.dtype}"

    psnr = compute_psnr(baseline_out, mlmodel_out)
    print("PSNR:", psnr)
    psnrs.append(psnr)

print("Mean:", np.average(psnrs))
print("Median:", np.median(psnrs))