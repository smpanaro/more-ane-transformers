# ml-ane-transformers

An attempt to convert a nanoGPT model to a CoreML-optimized version following the principles outlined in Apple's [ml-ane-transformers](https://github.com/apple/ml-ane-transformers/tree/main) repo and associated [article](https://machinelearning.apple.com/research/neural-engine-transformers). (Includes the various supporting/debugging scripts generated along the way.)

Ultimately, the model in ane_gpt2.py is both accurate and usable, however it is not as fast as the less-aggressively optimized version. This doesn't really make sense to me -- seems very likely that I messed something up, would love a PR if you see something.

From the repo root run `python -m src.ml_ane_transformers.convert-ane` to build a model. Edit `ane_gpt2.py` and the files in `ane/` to change the model.