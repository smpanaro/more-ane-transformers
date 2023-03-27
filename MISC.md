### March 25, 2023
Played around with the first working GPT2 float16 model (the one with only channel_mean as f32) to see how performance responded to a few changes. Completely ignores accuracy since that goes out with the window without the f32 mean.

This is the GPT2 512 length model with QK and K masks. Results from Xcode benchmark tool.

*all float16*
ComputeUnit.All = GPU 975, ANE 370, 283ms prediction, 940ms load, 402ms compilation
ComputeUnit.CPUAndNeuralEngine = CPU 8, ANE 1315, 534ms prediction, 634ms load, 408ms compilation

*all float16, macOS13 target*
Was curious if we picked up any optimizations with the newer OS. Doesn't seem like it.
ComputeUnit.All = GPU 975, ANE 370, 280ms prediction, 863ms load, 387ms compilation
ComputeUnit.CPUAndNeuralEngine = CPU 8, ANE 1315, 519ms prediction, 609ms load, 389ms compilation
takeaway: the GPU is fast

*float16, quantized to 4bit*
Quantization shrinks the weights on disk, but they're unquantized before any computation happens. Was curious if moving bytes from disk to ANE/GPU was a bottleneck. Not faster -- loading is actually (much) slower.
ComputeUnit.All = GPU 975, ANE 370, 283ms prediction, 1539ms load, 504ms compilation
ComputeUnit.CPUAndNeuralEngine = CPU 8, ANE 1315, 535ms prediction, 1182ms load, 434ms compilation

*nanoGPT f16*
Not optimized for Neural Engine. Seems to fail if you try to touch the GPU (weird), but pretty much flies otherwise. Multiple interesting learnings:
- Causal mask is all you need. k mask/attention mask adds nothing.
    - Should look at if there's a way to hardcode the triangular causal mask once instead of once per head (makes the proto file very big and conversion very slow).
- Returning a single token's predictions instead of all tokens' predictions seems to impact the speed dramatically (maybe just a coincidence, can't repro with the ANE-optimized model).
- Faster than the ANE-optimized network?! Do I have a bug?
ComputeUnit.CPUAndNeuralEngine = CPU 67, ANE 416, 224ms prediction, 3368ms load, 471ms compilation

### March 26, 2023
Have been playing with converting nanoGPT more. Tried gpt2-xl (1B+ params, had to use memory-optimized coremltools), not sure the PSNR (conversion script crashes there), but the results are ... interesting.
Prompt: What is the answer to life, the universe, and everything?
Answer: The answer to all these questions is Yes.\ntoolbar, toolbar\nドライブ
1215.84ms/token -- at 512 sequence length with no caching, this isnt' terrible?
MLModel loading/compilation is painfully slow. 8.5 minutes in my CLI (haven't tried Xcode profiler).

Decided that maybe my laptop is the problem. Ran both the un-optimized and optimized distilbert models from their ANE transformers [article](https://machinelearning.apple.com/research/neural-engine-transformers) on my Mac and the unoptimized one is faster on CPU than the optimized one on ANE. Maybe explains why they didn't use the M1 Max in their charts lol. So, going to compare GPT2 baseline v. ANE on my iPhone 12 Pro iOS 16.

*ANE optmized distilbert sst*
ComputeUnit.CPUAndNeuralEngine = CPU 606, ANE 4, 3.48ms prediction, 404ms load, 246ms compilation
ComputeUnit.CPU = CPU 670, 33ms prediction, 164ms load, 225ms compilation

*baseline distilbert sst (converted via exporters CLI, has attention mask input)*
ComputeUnit.CPUAndNeuralEngine = CPU 341, ANE 0 (!), 4.15ms prediction, 147ms load, 105ms compilation
(meta: coming back to this and I'm suspcious that I crossed the wires somehow and/or this model did nothing)

So seems like distilbert is faster on my phone too. Maybe there have been some upgrades between 16.0 and 16.3.1?

*baseline GPT (no qk mask, just using the hardcoded tril select op)*
ComputeUnit.CPUAndNeuralEngine = CPU 614, ANE 0, 2190ms prediction, 10672ms load, 693ms compilation

*ANE optmized GPT*
ComputeUnit.CPUAndNeuralEngine = CPU 1324, ANE 0, 9065ms prediction, 4686ms load, 755ms compilation
^ phone was pretty hot, might try again once cooler
ComputeUnit.All = CPU 0, GPU 1226, ANE 0, 9719ms prediction, 9818ms load, 619ms compilation

Re-running benchmarks on the baseline nanoGPT and GPU works today (9/10 times, probably what I saw yesterday). GPU alone is ~25ms and All (which ends up being GPU+ANE) is 45ms (361 ops ANE, 117 GPU). Not sure what to make of this. Going to try taking some models apart and comparing the pieces.

Ok, pulled down the ml-ane-transformers repo and trying to repro the results from those models. Noticed that the tests are only comparing the ANE-optimized model on CPU+GPU vs ALL. Tweaked them to compare with the non-omptimized distilbert model but they still report a speedup. Saved the models and ran the Xcode benchmarks for them.

*ref distilbert model aka unoptimized*
CPU_ONLY 262ms, CPU_GPU  9.45ms (all GPU), CPU_ANE 55ms (85% ANE)

*test distilbert model aka optimized*
CPU_ONLY 111ms, CPU_GPU 14.35ms (~all GPU), CPU_ANE 8.7ms (~all ANE)

So, it seems like: my GPU is very fast, and something is different about my GPT2 conversion.

Made some small tweaks to the 99% vanilla nanoGPT and got it down to 154ms. Mainly just moving the causal mask from f32 -> f16 (so it stays on the ANE instead of shifting to CPU) and using a tensor instead of a constant (also makes the mlpackage smaller). No clue if it's accurate though! Also seems finicky about running on the GPU. Weird.

Not accurate, bet it's the layer norm. Yup, PSNR jumped from ~50iirc to 67. Interestinglyy only added 30ms, looks like there's a CPU layernorm implementation.

Wondering if going to a larger model will let make me get away with f16 layernorm.
- Seems plausible. PSNR is higher with gpt2-large. Quality is there! 379-404ms on NE, CPU is sloooow (3s). GPU is still amazingly fast 130ms.