misc notes / explorations.
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

### March 28, 2023
Wondering if I somehow subtly messed up the ANE optimization. Taking the reference transformer model from ml-ane-transformers and slowly scaling it up and comparing performance.

|embed dim|enc layers|dec layers|heads|gpt2 approx|CPU+ANE (load)|CPU+GPU (load)|~# ops|
|--|--|--|--|--|--|--|--|
|768 |6|12|12|base  |92ms (0.98s)            |90ms (1.6s) |2.7k |
|1024|6|24|16|medium|243ms (3.2s)            |208ms (7.8s)|6k   |
|1280|6|36|20|large |443ms(7.5s)             |415ms(18s)  |10.3k|
|1600|6|48|25|xl    |7128ms (23.7s, CPU only)|720ms(40s)  |16.5k|
* all with 512 sequence length, # of ops differs by ~10 between ANE + GPU, all split einsum, all batch size 2 (oops)
* Also noticing the file sizes are lower here vs my other converted models. Maybe not 1:1.

One thing I notice is these are 100% on the ANE/GPU with no CPU. Wonder if shuffling data to CPU even for a couple ops is expensive. iirc the embedding layers run on CPU.
Large model is sooo slow to convert to CoreML. Just hangs in the MIL cleanup passes. Maybe # of ops? (1hr2m wow``) Running the Xcode benchmark for the largish model took a couple hours (despite allegedly being fast). Don't think Xcode is meant for this.

Kind of feels like smaller number of ops has an impact. Maybe we can get best of both worlds? Wonder why the ANE repo re-implemented FFN/LN/etc, mustn't be able to use the existing ones.


### March 29, 2023
Tried ENABLE_EXPERIMENTAL_PASSES -- doesn't seem like it makes a difference, the experimental passes don't seem like things that are in the base model. Maybe the ANE-optimized one?
gpt2-medium ANE: 246ms (233ms without experimental) -- pretty sure these were a no-op

Tried a swift CLI to rule out any Python wonk. Also used precompiled mlmodelc. Loading the larger models on ANE is still slow -- guessing that means the bottleneck isn't compiling, but loading things into ANE? aned is pinned at 100%

It's so much slower than the ANE-optimized reference transformer. Seems weird.

Watching the CPU monitor, it looks like aned bails for the xl model. Maybe that's why it's so atrociously slow? (5.8s/it) Yeah, CPUOnly loads ~immediately and is similarly slow to predict.

Trying an experiment where the embedding layers (wte, wtp) are removed. Think that's what causes the CPU to get involved.

### March 30, 2023
Remembered that the ml-stable-diffusion repo chunks the unet. Claims its for on-device deployment, but wondering if I see a performance difference on my mac.

|model|CPU+ANE(load time)|CPU+GPU(load time)|~ ANE #ops|
|stable-diffusion 1.4 unet (split ein, not chunked)     |726ms   (2.1s)|377ms  (2.9s)|3.5k (7CPU)|
|stable-diffusion 1.4 unet (split ein, 2 chunks, chunk1)|289ms  (657ms)|152ms  (1.1s)|1.6k (7CPU)|
|stable-diffusion 1.4 unet (split ein, 2 chunks, chunk2)|453ms  (565ms)|234ms  (1.3s)|1.9k (0CPU)|
|stable-diffusion 1.4 unet (original, not chunked)      |1040ms (894ms)|280ms  (2.9s)|1.9k (7CPU)|
|stable-diffusion 1.4 unet (original, 2 chunks, chunk1) |369ms  (328ms)|110ms (912ms)|915  (7CPU)|
|stable-diffusion 1.4 unet (original, 2 chunks, chunk2) |574ms  (299ms)|179ms  (1.2s)|1k   (0CPU)|

Seems consistent unfortunately.

Made a few dumb test nets: Linear -> gelu -> Linear with a ~gp2-xl amount of params (1, 512, 1600). After a certain size (somewhere between 2 and 3 GB), running with CPUAndNE gets you only CPU. Doing a similar test but with small weights and many ops. 500 loops of that linear/gelu/linear absolutely flies. More and more convinced it's a memory issue.

Tried to load 2 models at once (in Python, so not sure what device they landed on) and got `LLVM ERROR: IO failure on output stream: No space left on device`. Probably need to run + retrict it to ANE/GPU. Ah, just trying to load it on CPU+GPU throws that error now. Oops. Double ah, I was loading the too big for ANE model. Trying the one that fit on the ANE...

Seems like I'm able to load 4x 800MB models and get them to run on ANE (Python memory is ~1GB and I explicitly ran one as CPU_ONLY). Only question is if it's doing some clever de-duping since they're all identical. A question for tomorrow (trying to not get my hopes up but this + the pipeline code coming in the next version of coremltools...).

### March 31, 2023
4x800MB is the limit. The fifth gets put on CPU. This almost perfectly lines up with gpt2-xl (3.1GB).

Something interesting: I'm using 3 different sizes of models (2 each): 820MB, 860MB, 900MB. In that order, the 900MB ones were the ones that got put on CPU, so I put them first and the 820MB last. They load + predict one at a time on the ANE but fail to predict as a group. Maybe 900MB is a hair too big? Nope, same issue if I use all 820MB models.

But something else interesting: I can run the ml-stable-diffusion CLI with 3 copies of the unet (1.7GB) being predicted on back to back. Possible it's being smart and reusing the same one though. Also noticed that the second load of a model on ANE is much much faster (150s -> 2.5s) -- don't see that in Python so maybe worth a shot? Even just for that.

Sort of worked. Can load 6 800MB models at once, and they load fast after the first time (250-300ms each). Predicting on them back to back still has the same issue where the last 2 get stuck on CPU (can see it slow down). At least there's no more "Unable to compute NN output" errors.

Seems like there's a hard limit (~3GB), so let's see what we can do within that boundary. Didn't think it would work, but quantized all 800MB models (2.5GB on disk) to 4bit and still see the same performance trend (first 4 loaded models are fast/ANE, last 2 are slower/CPU). Quantized models load much slower (2s) too.

### April 1, 2023
- Memory limit of ~3GB
- Quantization doesn't help, actually makes it worse
- Prediction speed isn't amazing, but doesn't matter since the whole model doesn't fit
Basically need less float16s. So either something that removes (not zeroes) parameters or somehow merges them.
Alternatively, need some way to trick/dance with the ANE to load/unload parts of models on the fly (preferably fast).

First a really dumb test. Copied the SD unet mlmodelc 3x and seeing if they all run on ANE still. Still really fast. Maybe I am doing something wrong? Nah, I can repro that same behavior by copying my test net 6x. The initial load is slow for each, but it seems like ANE is smart enough after that to dedupe them. Can easily load all 6 (4.8GB) and predict on them.

Ohh, just realized that there's more in the coremltools 6.3 release than just pipelines. Not sure if anything good.. seems like a big refactor of the optimization passes. Hard to tell if anything new.

Two avenues of interest: sparsification by dropping weights. Try pipelines in coremltools?

Pipelines: Easy enough to make. Running the coremlcompiler on them results in all the weights being copied into each model of the pipeline (6 models * 6 weight files each * 820MB = 29GB...oof). Hmm.. it's fast. Like suspiciously fast. 3400ms which is 6x the ~600ms I saw for a single model running on ANE. Xcode can't profile it (30GB file lol not surprised). Looking at Instruments can see that in fact the first 4 models were on ANE and the last 2 on CPU. It's interesting, the initial prediction seems to be slower than subsequent ones. The 600ms I've been seeing with non-pipeline models is the first prediction only -- nope, went back and tried and its consistently slow. Pipeline must actually do stuff. CPU+NE no pipeline: 6s, pipeline: 1s. Not bad. 600ms might just be moving data to Swift and back? The 1s (and instruments trace) is acutally less than the Xcode profiles for a single 800MB net (90ms ANE, 1s CPU).

Suppose we give it a shot with gpt2-xl? Unexpected, but chunk szes of 1811MB and 1158MB seem to both run on the Neural Engine (!). I see 663.18ms/it with my test python CLI (gpt2-large is 393.95ms/it) and two distinct "Neural Engine Prediction" blocks when profiling with Instruments. Larger first chunk ends up with everything on the CPU so there is something up still. But this is great?

Something to look into. There's ~200ms of CPU happening in between ANE predictions. What is it? Wonder if it's something weird about Python. Maybe shuffling 512*50257 float32 (10MB :shrug:) is slow (ha).

(April 2) Looked closer and pretty sure it's that the lm_head linear layer gets done on the CPU (why??) and is just really slow. Can see in instruments that it's doing a bunch of BNNS inner products that take up almost the whole time. Some ideas for that, but later.

### April 2, 2023
Yeah... turns out that linear layer takes 30% of the whole prediction time. Selecting just the next token before lm_head actually keeps it in the ANE! Now xl runs in 450ms/token and ~all (except for the first couple ops) on ANE.

Couple options for what's next: sparsification (hard + not sure I fully grok), kv caching (hard + kinda grok), push pipelines to find the limit. The pythia family of models seems intriguing (there's a 2.7b instruct tuned variant on HF which would be cool).