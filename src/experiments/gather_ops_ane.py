import coremltools as ct
import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from os_signpost import Signposter
import os
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

"""
Test to see if any of the gather ops can run on the Neural Engine.

Result 9/27/23: Does not seem like it.
"""

T,E = 2, 4 #tokens, embedding size

@mb.program(input_specs=[mb.TensorSpec(shape=(T,1), dtype=types.int32),])
def gather_prog(indices):
    x = mb.const(val=np.array([[3,4,5,6,7], [8,9,10,11,12]], dtype=np.float32))
    indices_fp = mb.cast(x=indices, dtype="fp32")
    indices_clip = mb.clip(x=indices_fp, alpha=0., beta=1.)
    idx = mb.cast(x=indices_clip, dtype="int32")
    y = mb.gather_nd(x=x, indices=idx, name="y")
    return y

args = {
    "inputs":[
        ct.TensorType("indices", shape=ct.Shape(shape=(T,1)), dtype=np.int32),
    ],
    "outputs": [
        ct.TensorType("y", dtype=np.float32),
    ],
    "minimum_deployment_target": ct.target.iOS16,
    "compute_units": ct.ComputeUnit.CPU_AND_NE,
}
mlprog = ct.convert(
    gather_prog,
    convert_to="milinternal",
    **args,
)
print(mlprog)

mlmodel = ct.convert(mlprog, convert_to="mlprogram", **args)

inputs = {"indices": np.array([[1],[0]], dtype=np.int32)}
ml_y = mlmodel.predict(inputs)["y"]
print(ml_y)
mlmodel.save("gather_nd.mlpackage")