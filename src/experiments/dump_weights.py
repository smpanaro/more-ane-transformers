import coremltools as ct
import argparse
import glob
import re
import sys
import os
import numpy as np

from coremltools.libmilstoragepython import _BlobStorageReader as BlobReader
import coremltools as ct
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.frontend.milproto.helper import proto_to_types

"""
Parse out the ops that use weights from weight.bin from a .proto mlpackage proto file.
"""

parser = argparse.ArgumentParser(description='Parse .mlpackage to inspect weights')
parser.add_argument('model_path', help='path to a .mlmodelc file', type=str)
args = parser.parse_args()

def get_nn(spec):
    if spec.WhichOneof("Type") == "neuralNetwork":
        nn_spec = spec.neuralNetwork
    elif spec.WhichOneof("Type") in "neuralNetworkClassifier":
        nn_spec = spec.neuralNetworkClassifier
    elif spec.WhichOneof("Type") in "neuralNetworkRegressor":
        nn_spec = spec.neuralNetworkRegressor
    elif spec.WhichOneof("Type") in "mlProgram":
        nn_spec = spec.mlProgram
    else:
        raise ValueError(f"Invalid neural network specification for the model: {spec.WhichOneof('Type')}")
    return nn_spec

spec =  ct.utils.load_spec(args.model_path)
nn_spec = get_nn(spec)
# print(dir(nn_spec))
print(f"attributes: {nn_spec.attributes}\ndocString: {nn_spec.docString}\nversion: {nn_spec.version}")
print(f"{len(nn_spec.functions)} functions")
assert len(nn_spec.functions) == 1, "haven't seen a spec with > 1 function"
fn_name, fn = list(nn_spec.functions.items())[0]
print(f"{fn_name} function\n------")
print(f"opset: {fn.opset}")
print(f"attributes: {fn.attributes}")
print(f"block_specializations: {list(fn.block_specializations.keys())}")
block = fn.block_specializations[fn.opset]
print(f"{fn.opset} block\n------")
print(f"attributes: {block.attributes}")
print(f"operations count: {len(block.operations)}")

total_bits = 0
blob_reader = BlobReader(os.path.join(args.model_path, "Data/com.apple.CoreML/weights/weight.bin"))
for op in block.operations:
    for name, att in op.attributes.items():
        if att.WhichOneof("value") != "blobFileValue":
            continue

        valuetype = proto_to_types(att.type)
        is_tensor = types.is_tensor(valuetype)
        if not is_tensor:
            print(f"Skipping non-tensor type: {att.type.WhichOneof('type')}")
            continue

        if op.type in ["const"]: # constexpr too?
            # print(op.type, att.blobFileValue.fileName, att.blobFileValue.offset, op.outputs[0].name)
            offset = att.blobFileValue.offset
            dtype = valuetype if not is_tensor else valuetype.get_primitive()
            shape = () if not is_tensor else valuetype.get_shape()


            if dtype == types.uint8:
                value_bits = 8
            elif dtype == types.int8:
                value_bits = 8
            elif dtype == types.fp16:
                value_bits = 16
            elif dtype == types.fp32:
                value_bits = 32
            else:
                raise ValueError(f"Invalid dtype for blob file value type: {dtype}")

            num_values = np.product(shape)
            print(op.outputs[0].name, num_values, value_bits)
            total_bits += (num_values * value_bits)

            # if dtype == types.uint8:
            #     np_value = np.array(blob_reader.read_uint8_data(offset), np.uint8)
            # elif dtype == types.int8:
            #     np_value = np.array(blob_reader.read_int8_data(offset), np.int8)
            # elif dtype == types.fp16:
            #     np_value_uint16 = np.array(blob_reader.read_fp16_data(offset), np.uint16)
            #     np_value = np.frombuffer(np_value_uint16.tobytes(), np.float16)
            # elif dtype == types.fp32:
            #     np_value = np.array(blob_reader.read_float_data(offset), np.float32)
            # else:
            #     raise ValueError(f"Invalid dtype for blob file value type: {dtype}")

            # value = np_value
            # if dtype in (types.fp16, types.int8, types.uint8, types.uint32):
            #     value = np.frombuffer(value, types.nptype_from_builtin(dtype)).reshape(
            #         shape
            #     )
            # elif dtype == types.str and shape == ():
            #     value = str(value[0])
            # elif dtype in (types.fp32, types.str, types.bool, types.int32, types.int64):
            #     value = (
            #         np.array(value).astype(types.nptype_from_builtin(dtype)).reshape(shape)
            #     )
            # else:
            #     raise ValueError(f"Invalid dtype for tensor value: {dtype}")

            # print(value)

print(f"total: {total_bits} bits")