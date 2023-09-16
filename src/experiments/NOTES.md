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

### April 4, 2023
Trying pythia-2.8b using the chunked pipeline approach. It seems like only 2 of the 3 or 4 (depending on chunk size) models run on ANE. Can see all of them get loaded in the instruments trace, but maybe some fail? Wondering if it's the conversion to float32 at the end (why does ml-stable-diffusion do that?)

Profiled in Xcode each chunk individually and realized that 1, 3 and 4 all start with matrix_band_part that is not able to run on ANE. Hypothesis: chunk 1 runs on ANE because it sees that the rest of 1 and all of 2 are ANE. But when it gets to 3, it sees that it needs to switch between CPU/ANE then and for 4 and just opts for CPU the whole time. Pretty sure this is the causal mask (band part is a diagonal thing), so maybe can do something to avoid that op.

On the flipside, might be a coincidence, I'm pretty sure my test nets were 100% ANE and they still got pushed to CPU after a certain point.

Trawling the logs, some interesting things. There's definitely some mmap going on. Seems like that's why things go sideways (sometimes): "IODARTVMAllocatorGeneric::vmAlloc: VM exhausted".

Suppose we could try quantizing but I'm skeptical.

Not sure I'm reading vm_stat correctly, but it seems like both the pipeline + non-pipeline uses multiples of memory (5GB file -> 20GB non-pipeline, at least 40GB pipelined). Looks like we just run out? I thought the whole point was that you don't though.

Separately, figured I'd gander the logs to see if I can glean anything more about the non-pipeline model. See this from ANECompilerService:
Error: L2 read symbol does not exist in __OP___OP___OP___OP___OP_mlp_output_1_cast_decomp_1_1_decomp_1_0_nebypass_0_neconv
ANEC Compiler Input used legacy key names 'NetworkPlistName' 'NetworkPlistPath' - please update to use 'NetworkSourceFileName' 'NetworkSourcePath'

Could also try reading the MIL and/or generated proto to see if anything looks odd. The fact that it compiles to mlmodelc strangely is definitely sus (maybe worth an issue on coremltools?). (Update 4/5 filed and issue and they're looking.)

### April 5, 2023
Thinking quantization is worth another shot. Since NE mmaps, if it mmaps the quantized values (kind of sounds like it might) then we're probably in a great spot. Still gonna file some issues / radars.
Oof, chunk script doesn't work with the ops inserted by quantization.

Failed to load the quantized model, guessing its: "Illegal offset = -2144766976" from ANE Compiler Service
...coremltools/converters/mil/mil/passes/compression_passes.py:418: RuntimeWarning: invalid value encountered in divide
  params.quantized_data = np.round(val * (q_val_max - q_val_min) / (val_max - val_min)).astype(np_dtype)
...coremltools/converters/mil/mil/passes/compression_passes.py:418: RuntimeWarning: invalid value encountered in cast
  params.quantized_data = np.round(val * (q_val_max - q_val_min) / (val_max - val_min)).astype(np_dtype)

Noticed I have more free pages today, so trying the pipeline model again just in case. No luck.

Looking at https://github.com/eiln/ane/blob/624f5755eb9e7af55b3c4cce6f166c0ee27cc4d1/ane/src/ane.c#L635 which is a linux driver for ANE. It specifies the vm size as 0xe0000000 which is 3.758096384 GB. gpt2-xl is 3.28GB and pythia2.8b is 5.55 so that being the threshold seems plausible. However that code also implies that the T6000 (m1 pro) has 2 ANEs. Wonder how to confirm that. Ha, same person summarized them: https://gist.github.com/eiln/82406a83dc94019d7ffaed7d2e04f120 Seems I do have 2. `ioreg` only shows one though. Mine definitely lines up with the m1 pro in that table. Bummer. Yeah, sounds like the other ANEs are inoperable: https://news.ycombinator.com/item?id=30852818 - wonder if the M2s have a bigger vm_size (`ioreg -c AppleARMIODevice | grep dart-ane -A 20`).

Theoretically could squeeze 4bit 7B param model in there but that's it. And that's assuming that the translation happens at runtime. Pruning plus sparsity could maybe squeeze out a little more? Plus throwing the GPU in there (wonder what the limit is on that).

Was curious and did a sysdiagnose on my phone which dumped the ioreg. Interestingly, "vm-size" = <000000e0>, "ane-type" = <20000000> (<60000000> on my mac, which is type 96 so guessing that's 0x60, which is consistent with 0xe0000000 being 3.75GBish. 0x20 => 32 so lower than the 2020 M1 ~ seems about right).

Just spitballing, but maybe splitting the model up into chunks helped fit it in the available space? Guess the next thing is to get chunking working with quantization and see if that helps memory :crossed-fingers:

### April 6, 2023
Don't think the "Illegal offset = -2144766976" issue is due to the division. Get the same error with palettize/lookuptable option. Again, suspiciously close to 2^31. Weird to hit that with a 1.4GB file though, maybe something about how it's quantized?

Have been slowly chunking the 4bit quantized model smaller and smaller to see if there's a point where it runs on the ANE. At 7 chunks I hit the vmAlloc issue again. Fewer chunks (aka larger chunks) and only some would run on the ANE. Possible I'm generating corrupt models (I set force_replace to True during chunking to get past some errors) but they generate identical outputs to the non-chunked when run on CPU.

### April 7, 2023
Searching for the limits of ANE again. Chunked pythia-2.8b 4bit into 12 pieces (each ~100MB). As a pipeline, it fails with the generic "Error computing NN outputs". Tested each piece in Xcode and they are all 100% ANE except for the first and last, which is consistent with smaller models. Tried making smaller pipelines from the chunks and seeing where that falls over.
4,5,6 - 350ms 100% ANE, 350MB (~1.4GB dequant)
3,4,5,6,7,8 - failed in Xcode, 708MB (~2.8GB dequant)
1,2,3,4,5,6 - 627ms, from CLI all chunks ran on ANE with slivers of CPU in between, 704MB (~2.8GB)
1,2,3,4,5,6,7,8,9,10 - "Error computing NN outputs." from CLI, 1.18GB (~4.7GB dequant)
1,2,3,4,5,6,7,8,9 - 1000s, from CLI all chunks on ANE w/CPU in between, 1.06GB (~4.24GB dequant)
One wildcard still is the potential that the generated pipeliens are wrong, but that seems unlikely at this point.

Don't really know what to make of the 1-9 chunk pipeline. I wonder if some of the chunks are op and/or weight equivalent? Whoa.. what if those got cached by ANE? That doesn't make sense... obviously weights are different. Looking at the weights and they really are 1.06GB. Looking closer at the logs " IODARTMapper::iovmMapMemory: Map request failed (0xe00002bd) " -- suspicously right above the 0xe0000000 -- couple lines before VM exhausted.

Think I can explain the 4.24GB being > 3.75. The initial few ops in chunk 1 are actually fairly large (~250MB just for the gather). Wouldn't surprise me if that is the delta since those run on CPU not ANE.

So I think we're basically at the end of this road. Your model needs to be under 3.75GB (with a little breathing room) and even then you may need to split it into pieces. Larger than that and the hardware just doesn't support it.

Could give the manual pipelining another shot but my hunch is that you would have to actually unload/reload the models in ANE each time and I would be shocked if that wasn't on the order of seconds.

So for bigger models, seems like all ANE isn't a option. GPU is fast as blazes anyways, so maybe interesting to see if we can coax CoreML to use both? It doesn't seem willing to automatically if the model is too big for ANE. Maybe better to just start with GPU and look into KV caching?

Sidebar, thinking that the way to decode the ioreg vm-size is to flip the endian-ness. "vm-size" = <000000e0> => 0xE0000000 which lines up with that linux driver. If so, that means the M2 Air is "vm-size" = <0000ffff07000000> => 34.3GB(!). Will have to see.

### April 9, 2023
M2 Air
- gpt2-xl-pipeline: 406ms/it CPU+ANE, 1.3s/it CPU+GPU
- pythia6.9b: all CPU, would need to chunk
- pythia2.8b, no chunk: 2.3s/it CPU+GPU
- gpt2: 163ms/it CPU+ANE (could see some ANE usage in powermetrics, not sure how much), 118ms/it CPU+GPU
- pythia2.8b, 8 chunks: 1050ms/it CPU+ANE (100% ANE), 2.6s/it CPU+GPU

### April 10, 2023
2.8b runs 100% on ANE on M2. Cool. Seems like there's a spurt of CPU activity in between each chunk (probably doing the f32<->f16 conversion -- only a couple ms).

Tried 6.9b split into 8 chunks on M2 Air. Only had 16GB of RAM and it swapped to 16GB. It might've finished but I stopped it after 20-30 minutes. So, at a minimum, the M2 seems to be able to hold larger models than the M1 but unclear how much larger (ran out of time to try more).

### April 15, 2023
Have been experimenting with KV caching. Basic idea is to have a fixed size K + V matrix and use masking to cover the irrelevant positions. Flexible sizes don't work on the Neural Engine, so need to do it the harder way. (Should probably check that assumption though.) Currently have 2 models: gpt2-full (no caching) and gpt2-cached (with kv caching), both return kv_caches and only the cached takes one as input. Would be nice to have one model with conditional branching and share the weights -- hit a few issues going that direction so starting simple (disk space is cheaper than speed for now, could always force the full model onto CPU+GPU and leave ANE for cached).

It's a little hard to tell what hardware the cached version is running on (Xcode claims mostly ANE but ~30% CPU) but instruments makes it look all CPU (seems unlikely tbh). But either way the cached version is fast.. 25-30ms/it versus 70ms non-cached baseline (which is mostly ANE but some CPU).

Xcode profiles for gpt2:
cached: 29ms CPU+ANE (406/467 ANE), 35ms CPUOnly
full: 36ms CPU+ANE (414/421 ANE), 545ms CPUOnly (!) -- note this is all fp16 unlike the current benchmarks for gpt2 which have fp32 layer norms -- actually surprising that the results are intelligble at all... did I inadvertently fix a bug/break something in a new way?

The stitching of the new K/V with the cached K/V is tricky to do in an ANE-supported way. Currently using scatter which isn't supported. Trying to think of creative alternatives. Seems like slice and concat are also not supported.

Promising approach, build a static mask outside of the model and then use torch.where which works on ANE. Results are gibberish still but feels like an off-by-one. If so, base gpt2 runs on ANE again at 17ms (again all fp16). Fixed the gibberish (mask was inverted oops). gpt2-medium is 38ms cached (down from 103ms).

### April 16, 2023
Realized that the idea of branching between cached/non-cached models won't work with chunking without something really creative, since I don't think you can split in the middle of if/else. Should probably file a radar for the chunking thing.

Oof, cond (if/else) isn't supported on the Neural Engine. So you basically have to do your branching very high up (outside the model). Otherwise you have to go back to CPU to take the branch (you can see that it converts to f32 even).

pieline options:
x, cond -> normal -> cached -> out
=> can't share weights between cached and normal
=> can split them into chunks. complex but basically each chunk becomes a cond one branch that just forwards the input and a dummy output and the other that forwards the input and the real output. out selects the output

x, cond -> normal --> out
        -> cached -/
=> can share weights
=> can't split

Ideal solution is the Neural Engine allows bigger models.

Another option.. "smart" chunking. When splitting an if, walk both branches in parallel so that shared weights (assuming we figure out how to do that) stay together. Then at the end of the chunk, have two outputs (each branch provides empty versions of the other branch's). The start of the next chunk takes in the superset of outputs plus cond and continues either normal or cached branch picking the appropriate outputs. Complex, but:
=> shares weights
=> allows chunking

x, cond -> normal chunk1 --> (normal out1, cached out1) -> normal chunk2 --> out
        -> cached chunk1-/                              -> cached chunk2-/

Wish there was a way to do this in one linear pass...

Rough notes: one input... [1, n * X, hidden_size]
first num_layers*2*n are the kv_cache
next n are the kv_mask
remainder are the inputs -- need to see if split or chunk support remainders..
input.chunk(numlayers+numlayers+2, dim=1) retrieves them and is statically sized.
Fixed sizes are n = (powers of 2) + 1 and 2048 (or whatever the max seq is). First run will have huge matrices, but that's still almost certainly faster than encoding the inputs one at a time (I hope). You always take the KV cache path since even on the first pass it will overwrite the cached values. The KV mask is crucial for this too.
Embedding has to happen outside of the model as well (can pipeline it though, plus it's stuck on CPU anyways).
Dang.. using enumerated shapes makes the varied dimensions dynamic (instead of enumerating them like I assumed) so when you try to chunk them (which translates into a split) it fails to calculate the split size.
This angle still might be interesting when doing the branching model -- can at least enumerate many sizes then... probably (might hit more weirdness with the branching who knows).

Tried a gross hack by overriding how `chunk` is converted. Seems to work, but I bet it's risky and I don't know if it works on ANE.

### April 17, 2023
Going to change pace for a little to wait and see if there's any update on the weight sharing issue + clear my head. Read something about how ANE performance is less beneficial for small matrices, going to measure the impact on sequence size (e.g. how big the  K and V matrices are, all input tokens are 1). Maybe this path isn't even worth it?

Wow, using the model with the if-statement and CPU-only inference crawls. Oh and it doesn't work on Neural Engine yet, forgot about that. Hmm, even the non-branched is really slow today. Something sus. Trying to figure out what I changed, maybe making inputs f16? Yup.. that, also, interesting, I think I may be including the first prediction in the total and it looks like that is slower. Weird, the second time is the slowest for gpt2. But there is definitely a warm up period of 1-2 predictions.

All times are for a single input token, inferred min(196, seqlen) times and median time per inference. Not using the branched model.

CPU - No Cache -- added 9/9/23, measured once using Xcode.

|model      |seq length|CPU   |CPU+NE|NE - No Cache|CPU - No Cache|
|--|--|--|--|--|--|
|gpt        |1024      |79.6ms|40.3ms|             |
|gpt        |512       |43.2ms|23.7ms|69ms         |
|gpt        |256       |25.7ms|15.5ms|             |
|gpt        |128       |16.6ms|11.7ms|             |
|gpt2-medium|1024      |219ms |112ms|              |
|gpt2-medium|512       |116ms |62.9ms|103ms        |1.35s
|gpt2-medium|256       |67.9ms|40.1ms|             |
|gpt2-medium|128       |45.7ms|26.8ms|             |
|gpt2-medium|16        |26.2ms|20.4ms|             |
|gpt2-large |1024      |403ms |235ms |             |
|gpt2-large |512       |226ms |127ms |210ms        |
|gpt2-large |256       |138ms |78.4ms|             |
|gpt2-large |128       |94.7ms|53.7ms|             |
|gpt2-large |16        |55.2ms|37.5ms|             |
|gpt-xl     |512       |400ms |397ms*|455ms        |

\* consistent with this model not fitting on the Neural Engine.

Edit (4/18):
any prompt length, generate up to 512, gpt2-medium
single input, varying kv length len: (128*26.8) + (128*40) + (256*63) = 24,678ms = 48ms/token
single input, 512 kv length: 512*63 = 32,256ms = 63ms/token

any prompt length, generate up to 256, gpt2-medium
single input, varying kv length len: (128*26.8) + (128*40) = 8,550.4
single input, 512 kv length: 256*63 = 16,128ms
ideal state: 512, then single input, varying kv length

40 token prompt, generate up to 512, gpt2-medium
512 input, 512 kv length: (512-40+1) * 103ms = 48,719ms
512 then single input, 512 length: (1*103) + ((512-40+1)*63) = 29,902ms
ideal state: 512, then single input, varying kv length = 103 + ((128-40)*26.8) + (128*40) + (256*63) = 23,709ms

450 token prompt, generate up to 512, gpt2-medium:
512 input, 512 kv length: (512-450+1) * 103ms = 6,489 ms
512 then single inputs, 512 kv length: (1*103) + ((512-450+1)*63) = 4,072
ideal state: 512, then single input, varying kv length = 103 + ((512-450)*63) = 4,009ms

40 token prompt, generate up to 128, gpt2-medium:
512 input, 512 kv length: (128-40+1)*103 = 9,167ms
512 then single input, 512 kv length: (1*103) + (128-40)*62.9 = 5,638ms
ideal state: 512, then single input, varying kv length = 103 + ((128-40)*26.8) = 2,461.4ms

40 token prompt, generate up to 256, gpt2-medium:
512 input, 512 kv length: (256-40+1)*103 = 22,351ms
512 then single input, 512 kv length: (1*103) + (256-40)*62.9 = 13,689.4ms
ideal state: 512, then single input, varying kv length = 103 + ((128-40)*26.8) +(128*40) = 7,581.4ms

End Edit (4/18)

It seems to be an improvement over CoreML CPU but maybe not over a hand-tuned CPU implementation? Seems plausible -- there are 4bit quantized 7B param models that are 60ms/token.

Unrelated, was trying to narrow down where the PSNR falls off with pythia-410m. Thought it might be specific ops and was trying to find them experimentally:

PSNR with only these ops as f16:
layer_norm - 79
add+sub - 76
gelu - 94
softmax+const - 86
softmax - 86
const - 96
slice_by_index - 52 (wat)
mul - 80
linear - 66
matmul - 35
reshape - 47.5 (what)
concat - 34.8 (wat)
transpose - 50 wat
band_part - 96
gather - 82

Sets of ops as f16:
'band_part', 'const', 'gelu' - 91
'gelu', 'layer_norm', 'band_part', 'const', 'gather' - 77
'gather', 'const', 'band_part', 'gelu' - 86
'band_part', 'softmax', 'gelu', 'gather', 'const' - 52
'concat', 'band_part', 'gelu', 'const', 'gather' - 43
'slice_by_index', 'reshape', 'band_part', 'transpose', 'const', 'concat', 'gather', 'gelu' - 31

EDIT(May 7, 2023 - some of the "wat" can be explained by f32->f16 cast before those ops losing precision)

### April 19, 2023
Thinking about KV-caching again. A few options:
1. Branching model
Pros: Simpler, inputs / outputs are the same, fastest
Cons: Need to implement some weight sharing utility to avoid double storage(https://github.com/apple/coremltools/issues/1832), would need some creative solution to handle splitting large models into pipelines (probably a sequence of branching models would work).
Unknowns: Will it run on ANE since if/else is CPU-only?
2. Fixed KV cache size (e.g. max context length) and enumerated token inputs.
Pros: Simplest, should be supported, decent but not best speed (see math under table from 4/17, assuming that's right)
Cons: Not the fastest
Unknowns: Possible without branching?
3. Combined input
Pros: Fastest, if it works
Cons: Pipelining will be hard (have to recreate the combined input at each split), most complex, leaks out to the caler. Risk that it's not supported in the future.
Unknowns: Is splitting/catting tensors slow? Will it run on ANE?

Think that makes #2 the most appealing, since it would actually scale to xl and larger. Should open an issue about linked enumerated shapes (then #2 would be fastest + simplest). Any approach is better than what I'm doing now though.

Took a detour back to ANE-optimized model to see if anything stood out as weird. Doing some comparisons with the ANE-converted whisper models. Structurally, in Netron, they look like theyr'e doing the same transformations.

So.. the non-ANE whisper decoder model is slightly faster than the ANE-optimized one (12 vs 13.5ms). Not as dramatic as what I'm seeing with gpt2 but this decoder also has a sequence length of 1 -- could see that being impactful. ANE-encoder is 123ms, in line with what I see online. The non-ANE one does take an eternity to profile in Xcode...It comes in at 174ms (999869.21ms to load!!)

Reading a bit more, maybe the benefit of ANE is memory not speed? Seems silly to take such an extreme speed hit for memory/compilation time (especially since that is cached after the first run). Even in the ml-ane-transformers screenshot the compilation + load are slower. Still good be memory/power consumption -- I have a lot of memory so maybe I just don't see it?

Two things to look at: are there things in gpt2 but not the reference transformer (pos embeds, lm_head), what does performance look like for a single token?

### April 20, 2023
Whisper encoder is 176MB vs decoder 386MB. Maybe that's the delta? Too many weights to move around, not how fast the NE can go. Still seems weird that that would impact ANE-optimized more than non-optimized. gpt2 starts at 250MB.

Found an 18M param GPT2 on HF (smaller vocab too, about half).
ANE-optimized: 157ms prediction, 153ms load, 64ms compile
non-optimized: 8ms prediction, 63ms load, 38ms compile

Weird, noticed that the ANE mlpackages are noticeably larger. It's all the weight.bin. uh oh. The ANE-optimized has 15 extra weights (in the 18M param GPT). Trying to figure out why..

It's the LM head in the ANE transformer. It's 13MB of weights just for that. Seems like the non-ANE version is smart enough to reuse the weight? Ah, it's because I explicitly squeeze the weights so they're different. I now see that the whisper implementation uses an einsum. I wonder... Just wholesale removed the lm_head to see what happens. Wow 7ms... so just need to add that back in in a reasonable way. :facepalm: Adding that back in, I don't fully understand it but it comes out at 14ms. So still actually slower but I need to think more about the shape.

gpt2-medium is basically the same speed as non-optimized. Really need to check that the accuracy is correct before proceeding. Also splitting the model in 29 pieces at the end seems not-ideal.
[] Make sure the model is accurate.
[] Stop duplicating the embedding weights.
[x] Look at moving qk mask inside the model.
[x] Output mask before lm_head (seems faster).
~[] Try splitting at the beginning too (share the splits?)~ doesn't make sense
[x] Split the model less at the end. (doesn't seem to matter latency-wise)
[] Understand the mapping/packing of axes for ANE.
[x] Consider making qk a property instead of computed at run-time, see if that lets it get folded into a surrounding op. (it does not, but it does avoid some CPU computation at the start)
[x] figure out why the matmul einsum is not exactly equal to the conv2d (seems like there's some slight differences)

## April 23, 2023
I now remember that part of the benefit of switchin to the non-ANE version was the PSNR improved (pretty sure). So yes it's fast (seems like it should be faster but still) but it's not accurate.

So we've come full circle to when I started taking notes.

## April 24, 2023
Have been futzing around trying to squeeze precision out of float16. Tried Kahan summation as well as the KBN variant. Tried to take advantage of the fact that mean should reduce the magnitude which I think gives us a bit more precision, but no matter what I try it's always slightly less precise than the default ANE layer norm.

Was poking around coremltools and noticed that there's a MeanVarianceNormalizeLayerParams in the proto. Seems like it could actually be used to implement the ANE layer norm. I know the non-ANE gpt2 has a "layer_norm" op in the MIL. I don't quite see the connection but I think that compiles down to this MVN layer. Going to see if there's anything fancy we get by leveraging that layer. Initial tests make it seem identical to the ANE layer norm but it's a thread to pull. Actually looks like that NN thing is for NeuralNetworks not mlprogram (duh) -- but seems like even without reshaping, using mb.layer_norm works. So just how to use it with torch ...

Hacked it by putting a batch norm op in it's place and then replacing that with layer norm. The PSNR is nonsense (can try to measure that better tomorrow, makes sense batch norm != layer norm), but anecdotally the quality of the generations looks promising. But... it seems like the Neural Engine doesn't support this -- the generated text is absolute nonsense. Very fast of course.

Tried flipping the bias (a la the actual ANE LayerNorm). Good try but didn't help. Will have to file a bug/radar.

Should also try permuting the tensor.. maybe it's not so bad? 1024 x 512 isn't so big heh. Tried it... basically wrap the layer_norm with 2 transposes. Might have subtly messed something up, the quality is iffy but much better than gibberish. And it does run fast on the NE. This might be a good compromise? 80ms for gpt2-medium (75ms is the lowest I've seen it go ignoring accuracy) but it generates readable text. Should do a more empirical comparison/psnr but 20% improvement over the non-optimized gpt2 is pretty good.

## April 28, 2023
Two small experiments. Tried removing the nn.Embedding layer to see how slow that + the CPU->NE transition was. It was ~0.
Also tried increasing the number of splits in the mh_* attention arrays by 4x. Twice as slow for gpt2-medium (and I'm not sure if it even makes sense).

## May 7, 2023
Looked for low hanging speedups in coremltools. Upgrading the underlying protobuf version (which has performance speedups) didn't help. Seems like most of the time is spent in 2 optimization passes (casts and something else). Nothing obvious to me.

Switched tracks to look at pythia again. My recollection was that PSNR cratered for some of the models, but not all. Decided to look at some histograms of the weights. Sort of interesting. Smaller models have more extreme weights (-65/65 for 70m) but larger models (1b-2.8b) stay within the range of -10/10. Most weights are within (-2,2) for those larger models.

Revisiting experiments from 4/17 trying to find a culprit op. Realized that reshape being f16 should be no problem, so it must be that it loses precision for some op that comes after it (there are many...). With that lens, it seems like maybe matmul is the problem? Maybe the q@kT? Since f16 softmax was ok so probably not the @v.

Experimentally seems like that is the caes - q@kT in f16 exhibits the problem, but the @v does not. There are no other matmuls.

## May 8, 2023
Seems like the max value in the query grows as layers progress. Maybe it (silently?) overflows? Trying clamping it -- keeps the PSNR vs. HF high enough (>60) but doesn't help the CoreML model.

Trying a more targeted f32 test of just: k@qT matmul, the reshape and mask add after, softmax. Sort of helped - PSNR of 46 (vs 37). Lines up with the clamping of Q+K before matmul not really helping.

Maybe time to put a lid on it?

## May 9, 2023
Maybe >30 PSNR is ok? Would be interesting to compare the order of the logits -- maybe that is a better evaluation? Or maybe just the top N?

### May 10, 2023
Tried a few other measures of similarity (KL Div, Jaccard of the top k) and they all agree with PSNR. Jaccard is a little easier for me to intuit -- 90PSNR is 1 jaccard, 60PSNR is ~.95 jaccard, 35PSNR is ~0.84, 20PSNR is ~0.40. Think that's good enough to be usable (anecdotally it is, but this gives me a bit more confidence). Still feels odd to me that f16 would contribute so much error -- once the torch mps f16 issues are fixed would be interesting to compare and rule out CoreML oddity.

Turns out that is fixed in the nightly torch release. PSNR is 42 which is a smidge higher than CoreML. Possible I missed some float32 somewhere. The torch amp docs have a section for CUDA ops that are numerically unstable in float16, might be an interesting cross-reference.

### May 29, 2023
TIL zip has a max size of 4GB and macOS doesn't support zip64. New commands to compress + decompress large files:
compress: `tar -czvf - model.mlpackage | split -b 1800m -d -a 1 - model.mlpackage.tar.gz.`
decompress: `cat directory.tar.gz.* | tar -xzvf -`

### August 31, 2023
Heard back from Apple about my Radar for (B,C,1,S) layer norm. Turns out my repro case was overflowing float16. In retrospect, obvious. Though I'm confused about why I even filed it in the first place. What was I seeing?

Maybe I broken something when futzing with kv caching. 7d95b07 has convert-ane PSNR at 34 vs. 50 from earlier commits and <1 at 3470a25.

Maybe the right way to do this (instead of magicking away the batch_norm) is to update coremltools with a pass that fuses the pattern into a layer_norm op. TestFuseLayerNormOrInstanceNorm is close, can probably just modify it. Won't help with overflows of course.

## September 3, 2023
Modified coremltools 7.0b2 so it will replace the ml-ane-transformers layer norm with a proper layer_norm MIL op. Initial tests show that the performance is exactly the same. It definitely makes things easier to read in Netron, but that is not a super compelling reason. Maybe I'm missing something?

I also suspect that my current ANE-optimized GPT is somehow buggy (or maybe I just need to update the generate.py script). Edit: Still might be buggy, but it's at least consistent going back the last few commits.

Had a weird idea, I wonder if the MIL layer_norm axes=[-1] (wrapped with transposes) will yield the same results as layer_norm axes=[1] (no transposes). -- It seems like axes=[3] is more resilient to overflowing than axes[1] (weird, but I think it's true).

Might try doing the transpose wrapping based on an ENV var or something in coremltools.

## September 4, 2023
Remembered that iOS17 was touting using transformers. Curious to see if there's anything to learn from them.

Can see the kbd process loading:
loading resource id 0 type 46 with path /Library/Developer/CoreSimulator/Volumes/iOS_21A5326a/Library/Developer/CoreSimulator/Profiles/Runtimes/iOS 17.0.simruntime/Contents/Resources/RuntimeRoot/System/Library/LinguisticData/RequiredAssets_en.bundle/AssetData/en.lm/unilm.bundle/unilm_joint_cpu.espresso

Is this the same as http://codinfox.github.io/espresso/ ? https://orth.uk/apple-vision-framework/ seems to think so.


## September 8, 2023

Very rough notes from past few days of looking at models in Netron.

For the speech to text model:

Wp = workpiece, ph = phoneme? GPT’s guess but IDK
Wp seems to be the first transformer block
Wp cache is 12, 128,8,64 — num layers, seq Len, num heads, ??

I think it’s a sliding window KV cache maybe? KV cache gets split by layer and head until it’s 1,64,1,128 but then concat’d with something that 1,64,1,4. (So 64 is channels, last dim is sequence most likely here)

Next step after that is a slice by index, that pushes the last 4 elements out of the concerted value (so windowing so the length is 128)

Then we get to the matmul (einsum) which is 1,64,1,4 x 1,128,1,64 — so that does take advantage of the KV cache.

_does not answer how you get the KV cache in the first place_
Is it all zeros initially? Maybe that works? Since you shift in the new stuff to K before multiplying by Qt?

Eh.. I think you still are stuck. You have 1 degree of freedom so either you fix the size of K and vary input, or fix input and vary K (eh I’m not sure that makes sense)

Ok.. here’s an idea. Initial inference on CPU (!) give us the benefits of doing the whole prompt in one shot — take the KV cache from that and do token-by-token generation on ANE. Input is fixed at 1 token, but we can vary the size of the KV cache (and time scales proportionally with that .. so we will benefit).

This has the benefit of keeping the token → embedding bit in the model, versus my franken-input approach.

— Crazy idea, what if we had an enumerated shape for KV cache, and a range shape (default 1) for the inputs.

## September 9, 2023

Tried the enumerated + range. Works on CPU but not NE. Re-visited a few of my old ideas (splitting/slicing tensors) to try and cheat around the restriction of one enumerated shape and have roughly the same rsults as before: some work on CPU, some don't work at all.

Will have to try some on the new OS, but not today.

Inputs: (1,256) or (256,256) or (1,512) or (512,512) ((tokens, kv cache size))
Q dims: 1xC, 256xC, 1xC, 512xC
QKT dims: 1x1, 256x256, 1x1, 512x512
QKT+cache dims: 256x256, 256x256, 512x512, 512x512

I wonder if I can hand-tune an MLPackage that combines 1 set of weights with several different nets.

## September 10, 2023
Had a passing thought. What if we fixed the input tokens at 128 (not 1, but not huge) and then only varied the KV cache size. That could work. You would always recompute the last 128 tokens, but that seems to be a scale of things that is decently fast and it lets you infer large prompts in 128 token chunks instead of crawling through it 1 token at a time.

If you have < 128 tokens, you still compute them but KV mask them out.

Alas, it seems like the concat then slice doesn't work on ANE with enumerated shapes. Maybe on the new OS? Maybe there's a way to get coremltools to do so (since we can know at MIL-building time that the dynamic symbols are the same).

Maybe something like this: https://github.com/apple/coremltools/issues/764 (won't work exactly since I think you'll get a new symbol). Maybe torch.slice_scatter?

## September 11, 2023
Have a few too many balls in the air. Going to try to land some. First up is layer norm. Re-visited my test script (compare_layer_norm_axes) and confirmed that layer_norm(axes=[1]) yields the same results as ml-ane-transformers LayerNormANE. Will clean up my PR for that pass (doesn't seem to help performance but at least makes things easier to read).

Also confirmed that axes=[1] overflows before axes=[3]. Will need to clean probably create a new repro script for that, but will file a radar. In the meantime, nice to know that you can trade a bit of speed (15% ish) for precision if needed.

Next up (in no order):
- Lite clean up of my netron fork and push it (no PR though). (done)
- Look into sympy in coremltools and see if adding support for symbol +/- constant in slice/concat would be any different (hard to tell how that is used).
- Profile fixed-size KV cache implementation (128 input tokens, 512 KV cache) and see if performance is worth it (vs say 512 inputs always). If so, implement and re-build models. (Not as nice as if we could do enumerated inputs, but a smal step in that direction.)

## September 12, 2023
Put up a PR for the LayerNormANE in coremltools. Started to put together a radar (repro_layer_norm_axes) but found that axes=[1] and axes=[3] overflow at about the same point on Sonoma, so seems like that was improved! Would be interesting to see if there is still a speed difference between them at some point but the precision is more than welcome.

On a different note, the TinyStories peeps released a new 1.3B param model which might be interesting to implement (it's called phi-1.5).

## September 16, 2023
Cleaning up from the latest batch of experiments. Rought summary:

|File|Input Tokens|KV Cache|ANE?|
|--|--|--|--|
|enumerated_and_flexible_inputs|flexible|enumerated|no|
|enumerated_kv_cache|fixed|enumerated|no, but think it should|
|enumerated_shape_split \*|merged with ->|enumerated|no|

<sub>* enumerated_shapes_transformer, multi_variable_inputs are the same idea</sub>