import coremltools as ct

"""
Compose multiple test networks into a single pipeline to see if
it will do anything clever when running on the ANE.

** Requires coremltools 6.3 **
"""

models = [
    ct.models.MLModel(model_path, skip_model_load=True) for model_path in
        [
        # "test-net-20-loops.mlpackage",
        # "test-net-20-loops-2.mlpackage",
        # "test-net-20-loops-3.mlpackage",
        # "test-net-20-loops-4.mlpackage",
        # "test-net-21-loops.mlpackage",
        # "test-net-21-loops-2.mlpackage",
        "gpt2-xl_chunk1.mlpackage",
        "gpt2-xl_chunk2.mlpackage",
    ]
]


pipeline = ct.utils.make_pipeline(*models)
print(pipeline)
pipeline.save("gpt2-xl-pipeline.mlpackage")
