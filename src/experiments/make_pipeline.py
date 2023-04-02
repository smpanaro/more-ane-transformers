import coremltools as ct

"""
Compose multiple test networks into a single pipeline to see if
it will do anything clever when running on the ANE.

** Requires coremltools 6.3 **
"""

model_paths = [
    # "test-net-20-loops.mlpackage",
    # "test-net-20-loops-2.mlpackage",
    # "test-net-20-loops-3.mlpackage",
    # "test-net-20-loops-4.mlpackage",
    # "test-net-21-loops.mlpackage",
    # "test-net-21-loops-2.mlpackage",
    # "gpt2-xl_chunk1.mlpackage",
    # "gpt2-xl_chunk2.mlpackage",
    "gpt2-xl_2023_04_02-15_09_05_chunk1.mlpackage",
    "gpt2-xl_2023_04_02-15_09_05_chunk2.mlpackage",
]

models = [ct.models.MLModel(model_path, skip_model_load=True) for model_path in model_paths]

filename =  model_paths[0].replace("_chunk1", "-pipeline")

# Erroring here? You need coremltools >= 6.3 (which may not yet be released).
pipeline = ct.utils.make_pipeline(*models)
print(f"Saving pipeline model to {filename}")
pipeline.save(filename)
