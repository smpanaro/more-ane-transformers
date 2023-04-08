import coremltools as ct
import argparse
import os
import sys
import torch
from src.utils.psnr import compute_psnr

"""
Import a chunked pipeline model and an original model, perform predictions
on both and ensure that they are identical.
"""

parser = argparse.ArgumentParser(description='Load a pipeline CoreML modelpackage and compare it with it\'s non-pipeline version.')
parser.add_argument('pipeline_model_path', help='path to *-pipeline.mlpackage file', type=str)
# parser.add_argument('normal_model_path', help='path to non-pipelined *.mlpackage file', type=str)
args = parser.parse_args()

model_path = args.pipeline_model_path.replace("-pipeline", "")
# if os.path.exists(model_path):
#     print(f"non-pipelined model not found at {model_path}")
#     sys.exit(1)

pipeline_path = args.pipeline_model_path
# TODO: Need to add the model proxy for this.
# if os.path.exists(pipeline_path.replace('.mlpackage', '.mlmodelc')):
#     pipeline_path = pipeline_path.replace('.mlpackage', '.mlmodelc')
# if os.path.exists(model_path.replace('.mlpackage', '.mlmodelc')):
#     model_path = model_path.replace('.mlpackage', '.mlmodelc')

print(f"Loading pipeline from {pipeline_path}...")
pipeline = ct.models.MLModel(args.pipeline_model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
print(f"Loading normal from {model_path}...")
model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
print("Loaded both models.")

# input_ids = torch.rand((1,512,1600,), dtype=torch.float32)
input_ids = torch.randint(10000, (1,512,)).int()
output_mask = torch.tensor([2]).int()
print("input_ids.shape", input_ids.shape)
print("output_mask.shape", output_mask.shape)

po = pipeline.predict({"input_ids": input_ids, "output_mask": output_mask})["logits"]
print("Predicted on pipeline.")
mo = model.predict({"input_ids": input_ids, "output_mask": output_mask})["logits"]

print("psnr: ", compute_psnr(po, mo))
print("equal:",  torch.equal(torch.from_numpy(mo), torch.from_numpy(po)))