import coremltools as ct
import argparse
import glob
import re
import sys

from coremltools.libmilstoragepython import _BlobStorageReader as BlobReader

"""
Parse out the ops that use weights from weight.bin from a .mil file.
"""

parser = argparse.ArgumentParser(description='Parse .mil to inspect weights')
parser.add_argument('model_path', help='path to a .mlmodelc file', type=str)
args = parser.parse_args()

mil_path = f"{args.model_path}/model.mil"

weight_lines = []
with open(mil_path, 'r') as f:
    for line in f:
        if "BLOBFILE" in line:
            weight_lines.append(line)

weight_names = []
for l in weight_lines:
    name = re.search(r'>\s*(.*?)\s*=', l).group(1)
    weight_names.append(name)

for n in weight_names:
    print(n)
