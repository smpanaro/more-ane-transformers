import torch
import coremltools as ct
import numpy as np
import argparse
from stopwatch import Stopwatch
from collections import OrderedDict
import time

"""
Try loading multiple models at the same time and then predict on
them in sequence.
"""

compute_unit_by_name = OrderedDict([
    ("All", ct.ComputeUnit.ALL),
    ("CPUOnly", ct.ComputeUnit.CPU_ONLY),
    ("CPUAndGPU", ct.ComputeUnit.CPU_AND_GPU),
    ("CPUAndANE", ct.ComputeUnit.CPU_AND_NE),
])

all_models = [
    # "test-net-22-loops.mlpackage",
    # "test-net-22-loops-2.mlpackage",
    "test-net-21-loops.mlpackage",
    "test-net-21-loops-2.mlpackage",
    "test-net-20-loops.mlpackage",
    "test-net-20-loops-2.mlpackage",
    "test-net-20-loops-3.mlpackage",
    "test-net-20-loops-4.mlpackage",
]

parser = argparse.ArgumentParser(description='Load many models and predict on them in sequence.')
parser.add_argument('total_models', help='total models', type=int, choices=[x+1 for x in range(len(all_models))])
parser.add_argument('compute_unit', help='compute unit', type=str, choices=list(compute_unit_by_name.keys()), default="All")
args = parser.parse_args()

compute_unit = compute_unit_by_name[args.compute_unit]
print("Compute Unit:", args.compute_unit)

model_paths = all_models[:args.total_models]
print(f"Running with {len(model_paths)} models.")

input_ids = torch.rand((1,512,1600,), dtype=torch.float32)

models = []
load_times = []
initial_predict_times = []
group_predict_times = []

# Load each model and predict on it once.
for p in model_paths:
    print(f"Loading {p}")
    load_sw = Stopwatch(2)
    mlmodel = ct.models.MLModel(p, compute_units=compute_unit)
    load_sw.stop()
    load_times.append(load_sw.duration)

    models.append(mlmodel)

    inital_pred_sw = Stopwatch(2)
    mlmodel.predict({"input_ids": input_ids})
    inital_pred_sw.stop()
    initial_predict_times.append(inital_pred_sw.duration)

print("Sleeping...")
time.sleep(5)

print("Warm up group predict...")
for m in models:
    m.predict({"input_ids": input_ids})
for m in models:
    m.predict({"input_ids": input_ids})

print("Predicting back to back.")
group_predict_sw = Stopwatch(2)
for m in models:
    m.predict({"input_ids": input_ids})

group_predict_sw.stop()


print(f"{(np.average(load_times) * 1000):.2f}ms avg load time ({load_times})")
print(f"{(np.average(initial_predict_times) * 1000):.2f}ms avg initial predict ({initial_predict_times})")
print(f"{(group_predict_sw.duration * 1000):.2f}ms total group predict (avg: {1000*group_predict_sw.duration / float(len(models)):.2f}ms)")

