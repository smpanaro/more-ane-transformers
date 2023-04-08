import coremltools as ct
import argparse
import glob
import re
import sys

"""
Compose multiple test networks into a single pipeline to see if
it will do anything clever when running on the ANE.

** Requires coremltools 6.3 **
"""

parser = argparse.ArgumentParser(description='Stitch multiple chunked models into a single pipeline')
parser.add_argument('model_path', help='path to any of the chunked model .mlpackage files', type=str)
parser.add_argument('--range', help="comma-separated range of models to include", type=str)
args = parser.parse_args()

model_range = None
if args.range is not None:
    model_range = [int(x) for x in args.range.split(",")]
    assert len(model_range) == 2, f"range must have two elements: {model_range}"

def chunk_num(path):
    match = re.search(r'_chunk(\d+)\.mlpackage$', path)
    if match:
        return int(match.group(1))
    return None

pattern = re.sub(r'_chunk\d+', '*', args.model_path)
matching_files = [(chunk_num(path), path) for path in glob.glob(pattern)]
matching_files = [x for x in matching_files if x[0] is not None]
model_paths = [x[1] for x in sorted(matching_files, key=lambda x: x[0])]

output_filename =  model_paths[0].replace("_chunk1", "-pipeline")

if model_range is not None:
    # Shift left since chunks are 1-indexed.
    start, end = model_range[0]-1, model_range[1]
    model_paths = model_paths[start:end]

print("Will create a pipeline from the following models in this order:")
for mp in model_paths:
    print(mp)
correct = input("Is that correct? (y/n) ") in ["y", "Y"]

if correct:
    print("Creating pipeline.")
else:
    sys.exit(1)

models = [ct.models.MLModel(model_path, skip_model_load=True) for model_path in model_paths]

filename =  model_paths[0].replace("_chunk1", "-pipeline")
if model_range is not None:
    output_filename = output_filename.replace(".mlpackage", f"-{model_range[0]}to{model_range[1]}.mlpackage")

# Erroring here? You need coremltools >= 6.3
pipeline = ct.utils.make_pipeline(*models)
print(f"Saving pipeline model to {output_filename}")
pipeline.save(output_filename)
