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

Going to try to implement the fixed size KV cache.

params
seqlen=128
maxseqlen=512 (max is model dependent)
hidden_size=768 (model dependent)

inputs
input_ids=[1,seqlen] # padded on the LEFT
kv_cache=[num_layers,1,(maxseqlen-seqlen)*2,hidden_size]
kv_mask=[1,seqlen,maxseqlen] # really qk mask (should rename)
output_mask=[1] # index into seqlen, always set to seqlen-1 -- can get rid of but no reason to yet.

outputs
logits=[1,1,vocab_size]
new_kv_cache=[num_layers,1,(maxseqlen-seqlen)*2,hidden_size]

Strategy is to concat KV from cache with new KV from input to get the full KV (pretty standard iiuc).
To compute new_kv_cache slice off the leftmost seqlen elements.
input_ids will be padded on the left.

## September 17, 2023
Got it working as above with some small tweaks:

removed output_mask. Just plucks the last index always.
Added pos_offset=[1] which offsets the the positional embeddings so the first non-padding token gets index 0.

Initial results are promising. With seqlen=128,maxseqlen=512, gpt2 inference is ~50-54ms (Xcode vs CLI). Compared to without a cache (69ms), that's 22-25% faster.

Some profiles, using the CLI measurement in gpt2.py:

**Edit: Disregard this table, it was missing a key optimization.**
Context Size = maxseqlen
Input Size = seqlen
% is speedup compared to without a cache (inferring 512 seq len)
CLI = gpt2.py main function
|Model      |Context Size|Input Size|Median CLI (ms)| % |Median Xcode| %  |
|--         |--          |--        |--             |-- |--          |--
|gpt2       |512         |32        |40             |42 |--          |--
|gpt2       |512         |64        |44             |36 |--          |--
|gpt2       |512         |128       |54             |22 |50          |25
|gpt2       |512         |256       |73             |-5 |--          |--
|gpt2-medium|512         |32        |78             |24 |--          |--
|gpt2-medium|512         |64        |81             |21 |--          |--
|gpt2-medium|512         |128       |95             |8.7|91          |11.6
|gpt2-large |512         |128       |158            |25 |142         |32

Generally quite promising. As long as you're expecting more than a handful of tokens as output (probably the norm), these should provided a nice speedup.

Weird that gpt2-medium is an outlier. Vaguely recall seeing more ops on the CPU when run in Xcode. Going to do some profiling and see if I can get a breakdown of CPU vs ANE time.

Oh my. For gpt2-medium, 128 seqlen, 512 max, ~26ms is on the ANE out of 98ms total. Kind of hard to tell just from Instruments, but it kind of looks like a lot of time is spent just shuffling bytes between CPU<>NE. Not sure that we can make that go away. Actually, it looks like it's only ~9ms on the input side of the model, so probably similar on the output side which means the rest is CPU computations.

OH. Just remembered I moved the selecting of the last logit later than it needs to be (after ln_f+lm_head instead of before). Let's put that back. Wow, 95ms -> 52ms (49.5%!).

Output is ever so slightly different (missing a leading paren in the gpt2-medium output). Ugh, should debug, but for now going to let it ride. (Edit: This was caused by the [:, -1,:] slice instead of [:, [-1], :]).

Also, looking at the instruments trace. Looks much more reasonable. Still 9-10ms to pass the input to the ANE and about the same to pass it back out. There is a 2ms hop to CPU at the end, but not sure it's worth chasing down. So in that 50ms span, 20ms is spent just moving all the input bytes in and output bytes out (!).

Actually.. looking at Xcode, it looks like the inner product (tokens <> embeddings) is ~the only thing running on CPU now. So maybe there is a bit on both ends worth looking at (those tensors are large). Just need to do the split/concat trick to get them under 16384. (See [this](https://github.com/RobertRiachi/ANE-Optimized-Whisper-OpenAI/blob/d42252155b8e29b2e2c32e7b911ec647198547fb/model.py#L181C1-L181C1) note in a whisper conversion.)

With earlier logit plucking (caveat their might be a slight inaccuracy in the outputs -- see above), but without split/concat for embeddings:

|Model      |Context Size|Input Size|Median CLI (ms)| % |Median Xcode| %  |
|--         |--          |--        |--             |-- |--          |--
|gpt2       |512         |128       |18             |72 |--          |--
|gpt2-medium|512         |128       |52             |49 |--          |--
|gpt2-large |512         |128       |--             |-- |139         |33

gpt2-large segfaults when trying to load after conversion now. hmmm. Runs fine in Xcode though. Only thing that changed was moving the plucking of the last logit before ln_f+lm_head.

Chatted a bit with ChatGPT and came up with an idea for moving the intial nn.Embedding call to ANE. The idea is to split the embedding into N embeddings each with size < 16384. Then we take the input_ids and copy them for each split, masking out values that belong to other splits and shifting values that belong to this split so they are zero-indexed. Finally sum the results of all the splits (since the masked out values will be 0 valued).

Even gpt2-medium is segfaulting. Suppose this is better than only large segfaulting, but no clue what I changed. I think it's the difference between `x[:, -1, :]` and `x[:, [-1], :]`. Yeah, it's definitely that. Yikes. Good news though, it fixes the slight output error mentioned above.

Trying the split/concat trick just on the output (will need ^^ for input). It's actually 1-2ms slower for both gpt2 and gpt2-medium. Looks like it still hops to CPU and back, and actually adds an extra CPU hop at the end. Seems like two ops: one general_concat (one can run on ANE though and does not), a gather_nd before all the embedding inner_products. Maybe the latter is from the [-1]? In Instruments, it looks like the inner product at the start itself is fast. So just a question of if moving the 128*768 embeddings is slow. Maybe? But that's peanuts compared to the KV cache.

how many floats?
4,718,592 for gpt2-mediums KV cache
65,536 for qk mask (could compute this in-model probably)
131,072 for the embeddings. eh.

Unrelated: It occurrs to me that Xcode profiler's list of ops is straight-up the espresso plan. Would be nice to get a hand on that for real + visualize it.

## September 27, 2023
Doesn't seem like gather will run on ANE, no matter the size.

Splitting the final lm_head to fit on ANE actually makes gpt2 slightly slower. Will see if that holds for larger models.

gpt2-large w/ splitting:
104ms median on the command line
An example prediction:
`|[CPU 17ms][ANE 55ms][CPU 2ms][ANE 2ms][CPU 25ms]|`

w/out:
101ms median on the CLI
`|[CPU 17ms][ANE 53ms][CPU 6ms][ANE 2ms][CPU 25ms]|`

## September 28, 2023
gpt2-medium spends 20 of 50ms in the leading/trailing CPU sections. gpt2-large spends 42 of 103ms there. Both are ~40%.

Looking back at how Apple does it. Their models starts like this:

input (1x128x1x3)
|
V
split_nd -(1x128x1x1)-> inner_product(nB=256,  nC=512) -(1x128x1x512)-> add -> add ->
        |-(1x128x1x1)-> inner_product(nB=15000,nC=512) -(1x128x1x512)---^      ^
        |-(1x128x1x1)-> inner_product(nB=2    ,nC=512) -(1x128x1x512)----------|
(all 3 inner products have is_lookup=1, so maybe that's just how gather compiles down to espresso)

Their output is simple:
logits(1x1x128x512) -> inner_product(nB=512,nC=15000) -> softmax_nd(axis=-1) -> out(1x1x128x15000)

Ooooh, opening the Xcode performance report in Instruments gives more fine-grained information.

For gpt2-large:
89ms prediction total (in Xcode)
3ms, 1us, 7us, 700ns Input Copy (guessing that is kv cache, pos_offset, input_ids, qk_mask -- based on sizes)
11ms, 23us 'Neural Engine - Data Copy'
400us 'CPU' compute (included in this is 104us of Neural Engine Data Copy)
55ms Neural Engine Request
2ms 'CPU' compute
4ms Neural Engine Request
22us, 12ms Neural Engine - Data Copy (2 outputs: logits, kv_cache)

I wonder if any of the overhead is float32<>float16. Otherwise not sure what to tune here. Seems like this is the trade-off? Shuffling huge tensors instead of doing the computation.

gpt2-large float16 inputs and outputs:
110 ms median CLI
62ms in Xcode
'Neural Engine - Data Copy' basically goes away

Two takeaways from this:
- Casting floats is slow.
- There is some meaningful bottleneck in the Python code now.

Ahhh, coremltools casts all float16 to float32 (on both input and output) because C++ doesn't have a float16 type. Seems like there are some workarounds in PyBind11. The speedup is probably worth at least a look.

Updated it so the output isn't casted. But that means it has to get casted on the way in (since KV cache is passed back in). So now the end conversion is fast, but the start conversion is even slower. Added a conversion at the input end too. Still not as fast as Xcode but took gpt2 from 20ms to 17ms. (Xcode is like 10.5ms). Takes gpt2-large to 91ms on the CLI (from 110ms). Roughly consistent with ~15% improvement for gpt2.

Can't really tell if the remaining ~30% is an artifact of Python<>ObjC or if it's something that the Xcode performance tool doesn't measure. Instruments makes it look like it's all copying/moving buffers.

Easiest thing is probably to write a swift CLI (or see if the one I wrote previously works still).

## September 30, 2023
Reconsidered the Swift CLI approach for now. Don't really want to translate all of the input building logic into Swift.

Comparing the Xcode instruments trace with the CLI.
CLI spends most of CPU time in these 2 cousins:
11.00 ms   22.4%	0 s	 	                                           EspressoLight::espresso_plan::__copy_inputs(std::__1::shared_ptr<EspressoLight::plan_task_t>, std::__1::shared_ptr<Espresso::abstract_batch> const&, int, std::__1::shared_ptr<Espresso::net>)
11.00 ms   22.4%	11.00 ms	 	                                           bool CoreML::vectorizeMultiArray<_Float16, float>(CoreML::MultiArrayBuffer const&, CoreML::StorageOrder, CoreML::MultiArrayBuffer&)

The Xcode trace does have 1 span of converting float to float16, but only at the beginning of the trace. Maybe it's reused? Hope not.

Looking specifically at the CLI trace, it spends 6ms in PyArray_NewCopy at the end. Maybe can get rid of that by providing an output buffer? 6ms (out of 90ishms) is pretty small though.
The bigger chunk is the 2 cousins from the beginning, as mentioned above. Maybe there's some way to provide input features that is better?

Would be interesting to use the "provide your own output buffer" for KV cache values, and see if you can just hand that back straight in to the model. Maybe speed up both output and input. The MLPredictionOptions header has good comments (float16 MLMultiArray with a CVPixelBuffer backing sounds interesting).

"Use this initializer to create IOSurface backed MLMultiArray, which can reduce the inference latency by avoiding the buffer copy." in the header for init a MLMultiArray with a buffer. Tried this with an blank buffer, and it is definitely faster. Still a slight bit of overhead compared to the Xcode profile. Need to figure out how to get they numpy data into the buffer now.

Got the data into the buffer. (Well, the generated text is correct. Possible I'm not being entirely safe with memory of course.) Am approaching the Xcode profiler's performance. For gpt2 it is 10.5ms and CLI median is now 14ms. The delta seems to be purely the memcpy (assuming that's _platform_memmove) going from numpy <> C.

gpt2-large comes down to 80ms on the CLI (vs 62 in Xcode).

Think I might have a memory issue (as kind of expected...). gpt2-medium output doesn't look quite right.

So next steps:
- track down memory corruption (or confirm it's ok)
- comment in MLPredictionOptions.h "For the best performance, use page-aligned address in MLMultiArray." -- try page-aligning the input buffer
- See if there's a way to get the CoreML instrument to run for the CLI.

Poking around the console logs, trying to find out how to get the Core ML instrument to work. This is interesting:

"Kernel validation warning tok_emb_cast (inner_product) @ 7: Inner product (is_lookup = True) requires nB to be bounded by 32768. Experienced compilation failures for larger factors in testing."
Relevant MIL:
tensor<fp16, [50257, 768]> transformer_wte_weight_to_fp16 = const()[name = tensor<string, []>("transformer_wte_weight_to_fp16"), val = tensor<fp16, [50257, 768]>(BLOBFILE(path = tensor<string, []>("@model_path/weights/weight.bin"), offset = tensor<uint64, []>(64)))];
tensor<fp16, [1, 128, 768]> tok_emb_cast = gather(axis = tok_emb_axis_0, batch_dims = tok_emb_batch_dims_0, indices = input_ids, x = transformer_wte_weight_to_fp16)[name = tensor<string, []>("tok_emb_cast")];

"Kernel validation warning op_1539_cast (inner_product) @ 524: Output blob dimensions exceed ANE limit."
MIL:
tensor<fp16, [1, 1, 50257]> logits = linear(bias = var_1539_bias_0_to_fp16, weight = transformer_wte_weight_to_fp16, x = input_cast)[name = tensor<string, []>("op_1539_cast")];

Cool to finally be able to match these concretely up between Xcode and the MIL.

This is how the Xcode runs the instrument.
xctrace record --template Core ML --output /var/folders/7p/rj7zj_q13qzg96rw0081xslr0000gn/T/perftab_60254D8E-4F9B-40AD-BC39-DA034FD36293.trace --launch -- com.apple.coremlprofiler 835A2AD4-6450-431F-AF5D-EC26581B70DB

## October 1, 2023
Set up a Swift app and was able to attain the same performance as the Xcode profiler (using dummy inputs). Interestingly running with ComputeUnit.All was slower than CPUAndNE. Needed to configure the CVPixelBuffer inputs to get good performance, so tried the output backings too -- no effect.

Some interesting symbols:
MLModelConfiguration - (void)setAllowsInstrumentation:(bool)arg1;
MLModel - (void)enableInstrumentsTracing;

#1	0x00000001987fac08 in __MLLoggingGetInstrumentsActiveChannel_block_invoke ()
^ is where os_log_create("com.apple.coreml", "DynamicTracing") is called

MLLoggingAllowsInstrumentation(bool ifTheConfigHasAllowsInstrumentation, char* pathToMlmodelc)

Spent way too long trying to figure out why the CoreML instrument isn't working for my CLI runs.tldr of what I learned: the CoreML instrument is powered by signposts in the "com.apple.coreml" subsystem with the DynamicTracing category. That category is only saved when Instruments is attached. The Instruments "CoreML Package" (in Instruments' settings) transforms these signposts into well-formed data that is used to populate the CoreML instrument in traces. It /seems/ like something in CoreML is preventing those from coming through for my Python CLI -- I'm able to use the python signpost package to manually record DynamicTracing category signposts that show up.

## October 2, 2023
Realized I can add my own signposts to coremltools. Wrapped both the input features and output features conversion. Provides a nice separation from when control is handed to CoreML. Confirms that the slow memcpy is in my coremltools code and not CoreML. Tried building coremltools in release mode (I had manually set it to debug oops) to see if that helps -- does not seem like it.

The CPU chunk in between the two ANE inferences has been bugging me. Went back to using the split lm_head, but moved the KV cache stack before it. The stack runs on ANE but I _think_ since it was after the split lm_heads cat, we were paying some extra context transfer cost. By putting the stack before the split head (the order doesn't matter), the ANE inference is one solid chunk. ~The same speed for gpt2, but takes gpt2-large to a median of 73ms (last best was 80ms) on CLI and 56ms in the Xcode profiler.

Looking at the total durations for gpt2-large. 7.43s spent in predict, 1.76s spent converting inputs/outputs. Beyond something egregious in the model itself, seems like optimizing the inputs/outputs is the last thing to squeeze.

(I wonder if taking all these learnings back to the ANE-optimized GPT would be beneficial.)

Reading a bunch about how CVPixelBuffer + alignment work. Wondering if that might explain some of the "corruption-y" looking things I was seeing. tldr is CVPixelBufferGetBytesPerRow() != width necessarily.

## October 3, 2023
Had an idea while I slept for optimizing the inputs/outputs. Added signposts and confirmed the delay is almost entirely the KV cache. The hiccup is that we have to take that big chunk of memory from CVPixelBuffer to Python and then back for the next prediction. We can add a cache for those values inside the ObjC bridge and not even bother sending them to Python. Something a la:

py::dict predict(const py::dict& input, const py::dict& options) const;
where options = {
        # cached_key_name is saved from the output (and not returned)
        # if cached_key_name is saved for this model, it is added to the inputs
        "cached_input_key_mapping": {
                "input_key_name": "cached_key_name",
        },
        "ignore_cache": True, # for the first prediction
}

Don't think this would get accepted into coremltools though. Feels a little to hacky/application-specific.

Also, tried the page-aligned MultiArray(dataPointer:...) in Swift for the output buffer. It's slower, despite saying "for best performance". Seems reasonable since IOSurface looks to incur ~no copies.

Last two stones to turn over:
- get to the bottom of if there is output corruption and why
- go back one last time and see if splitting the input tok_emb weights is possible to get on ANE

Looking at the latter (more interesting, sue me) and noticed interesting logs from com.apple.CoreAnalytics:
"Dropping "com.apple.CoreML.MLLoader" as it isn't used in any transform (not in the config or budgeted?)".
Dropping "com.apple.CoreML.MLPrediction" as it isn't used in any transform (not in the config or budgeted?)
Those sound a lot like the events that are missing from the CoreML instrument (model load + prediction stats).

## October 4, 2023
I feel like it might be possible to run the embedding layer (gather) on ANE, but I can't figure out how. I tried to mimic the input to the keyboard LM, but I can't figure out how to arrive at equivalent an espresso net.

Added an assert to see if the CVPixelBuffer was getting padded with extra row bytes (coincidentally all the gpt2 dimensions are divisible by 16, so maybe moot). Has not fired yet. Also experimented with using vImageCopyBuffer (noticed it was used for translating PIL images to CVPixelBuffer). Does not seem to provide a speedup, but is nicer than having to manually handle potential padding issues.

Tested generation on vanilla 7.0b2 for all 3 models and compared with my local changes (float16, CVPixelBuffer, vImage). The generation (argmax) output is the same. So that's probably enough to rule out corruption. Will keep my eyes peeled though.

## October 5, 2023
Reviewing my notes, and some of the times I have for models early on are much higher than I remember. I should really come back and try the ANE-optimized models again. Not today though.

Was skim-watching a WWDC talk about profiling pytorch MPS. They have some signposts in there for profiling. Might be interesting to peek and see how it works. Would be so nice if there was something like that for ANE.

Tried my caching idea (keep a cache of the kv_cache CVPixelBuffers in CoreMLPython and feed the prior output into the next input). It works great. Am seeing median CLI times of 11.5ms for gpt2 (vs 10.3ms Xcode). gpt2-large is 56.6ms CLI (vs 55.9ms Xcode). Don't think the last ~1ms (which seems like a constant) is worth chasing. Does feel like the Python overhead is near a minimum now -- pretty sweet.

## October 6, 2023
Got the CoreML instrument working with a plain ObjC++ CLI. The binary needs to have the get-task-allow entitlement. I'm guessing if I signed python with that it would work. However, I don't actually need that since I've added my own signposts to coremltools.

## October 9, 2023
Was curious and tried running gpt2-large in Xcode 15 on Sonoma. Hoped it would be a bit faster, but it does a bunch of CPU casting/memcpy and is like twice as slow. Something for later.

## October 10, 2023
Converted gpt2-xl, since I haven't yet seen what it looks like. Seems like there's a bug in coremltools 7.0b2, or in my code, that causes the pipeline stitching to not work. Still works on 6.3 -- will have to look. Also, noticed that the intermediary outputs are all float32. Should see what it takes to convert them to float16.

Had to jump through some hoops. Need to chunk on 7.0b2, then make pipeline on 6.3. And also need to make the intermediary outputs float16, otherwise it won't run on the ANE. But it does run, and the output looks sensible. Median 121ms/it. Wowowow. Proportionally on-par with the other models, but still something since it actually feels usable now.

## October 12, 2023
With Quinn's help on the developer forums, figured out how to get the CoreML instrument working with the Python CLI. Need to re-sign the Python binary (which is a bit hidden) and give it the get-task-allow entitlement.
codesign -f -s - /opt/homebrew/Cellar/python@3.10/3.10.12_1/Frameworks/Python.framework/Versions/3.10/Resources/Python.app/Contents/MacOS/Python --entitlements entitlements.xml
https://developer.apple.com/forums/thread/739159?login=true&page=1#768381022

Was thinking it'd be interesting to chunk up the transformer into a pipeline and see which parts in particular are slow. coremltools' `extract_submodel` seems like it would make it easy enough.

## October 18, 2023
Fixed a off-by-one bug in the KV cache creation that would cause anything that relied on it to be wrong.

Read an interesting [post](https://kipp.ly/transformer-inference-arithmetic/#kv-cache) that says there is an ideal balance between memory + compute boundedness. Specifically that there's a threshold to KV cache sequence length where memory bandwidth becomes the bottleneck and smaller query sizes don't matter. Figure that would be interesting to probe experimentally. Since we can only have 1 input sequence size, picking an optimal one would be beneficial.


|model      |input length|median time (ms)|
|--         |            |                |
|gpt2\*     |256         |19.9            |
|gpt2\*     |192         |14.4            |
|gpt2\*     |128         |10.9            |
|gpt2\*     |96          |10.2            |
|gpt2\*     |64          |8.5             |
|gpt2\*     |32          |7.6             |
|gpt2\*     |16          |6.9             |
|gpt2-large |256         |CoreML errors   |
|gpt2-large |192         |91.3            |
|gpt2-large |128         |57              |
|gpt2-large |96          |53.5            |
|gpt2-large |64          |45              |
|gpt2-large |32          |41.1            |
|gpt2-large |16          |36.7            |


<sub>* no CPU ops, so accuracy suffers</sub>
<sub>all times are median of 200 inferences, M1 Max</sub>

## October 20, 2023
Realized my KV cache implementation is not awesome for initial prompts > 128. It will ignore everything but the last 128 tokens as implemented. Would be easy-ish to adjust it to infer the first 128 and then 129+ one at a time but that would be painfully slow. Right now I shift the KV Cache by 1 token and then slice off the rightmost 127 columns. e.g.
```
# T = 2, input sequence
# k = [o,o,o,o,o,o][n,n] # k = old_k + new_k
#       [o,o,o,o,o,n] # output cache
```
Two potential solutions:
- output the full cache and roll it either by 1, or in chunks of 128 until the full initial prompt is consumed
- try to pass a new input and do a dynamic roll + slice in the model

The first option means shuffling a decent amount more data and also makes it harder to keep the cache out of Python code (ObjC<>Python is sloow).
The second probably won't run on the ANE, but would make things easier outside the model.

Will try option 2. Think changing the shape of the cache from [layer, batch, 2*seq, dim] to [layer, batch, seq, 2*dim] will make slicing easier.

## October 22, 2023
~Maybe a third option for caching that is simpler: pad on the right and always slide window by input length.~ edit: Doesn't work.

Basically need to have this input cache (simplified to just columns):
[1 2 3 4 5 6 7 8 a b c d]
Work for either:
[b c d e] (single new token)
[e f g h] (batch of new tokens)

input_ids: 128
input_length: 1-inf. - length of full sequence even parts that are > 512
~new_tokens_length: 1-128 - length of tokens in input_ids that are new~
next_new_input_length: 1-128 (default 1) - length of new tokens in next input_ids

input_ids: [b c d e]
input_length: 5
new_tokens_length: 1
old_k+new_k = [5 6 7 8 a b c d]

input_ids: [e f g h]
input_length: 8
new_tokens_length: 4
old_k+new_k = [5 6 7 8 a b c d]

Tried it. Unfortunately, seems like this requires dynamic slicing (although technically things end up a static size) and it gets forced to CPU.

## October 25, 2023
Tried a few different things to get flexibility without leaving the ANE. It seems like as soon as you introduce anything dynamic -- either a dynamic shape, or even a static shape but take from a dynamic offset -- you get moved to CPU. This tends to be pretty slow, especially if you have ops that require conversion to float32 for CPU (some of the slice ops seem to). Roughly doubles duration (11->20ms) for gpt2, and it's proportional to KV cache size so larger models would be worse.

Three main things I tried:
- tensor slicing with a dynamic offset: full_kv_cache[:, :, 1d_index_tensor:1d_index_tensor+384, :]
  - this translates to slice_by_index in MIL and runs on CPU
  - seems to be some correctness issues (maybe quirks of float16 to float32 and back?)
- full_kv_cache.index_select(2, torch.add(1d_index_tensor, torch.arange(384, dtype=torch.int32)))
  - this translates to a gather in MIL and runs on CPU
- Custom support for narrow (there's a GH issue with it) torch.narrow(full_kv_cache, 2, 1d_index_tensor, 384)
  - translated this to slice_by_size, thought that the static size+shape might help, but this still runs on CPU (the conversion ends up doing something similar to slice_by_index and building a dynamic index tensor using concat) and requires conversion to float32

Based on this a few ideas:
- return two outputs: one cache assuming that the whole 128 inputs were consumed, one cache assuming that only 1 new token was consumed.
  - pros: guessing this will be fast, since I think it will sidestep memory copies (hopefully)
  - pros: really think this will run on ANE
  - cons: chance that returning two offset copies of the same tensor causes a copy or something funky
  - cons: have to switch between small + big cache which is a bit of a headache esp. in my custom coremltools implementation
  - ~cons: slightly more complex since chunking the initial prompt needs to end on a multiple of 128 (first prompt has to be padded which is a bit unintuitive)
  - ~cons: caches will have to be full size (max seq length), hopefully zero impact if using pixel buffers
- revisit enumerated inputs.
  - pros: if they all ran on ANE, this would be maximum speed.
  - cons: need to figure out qk_mask here. possible that using a broadcastable mask would work since we only care about the last token.
  - cons: if we ever cared about anything other than the last token, this would not work.
  - cons: last time I tried enumerated inputs it did not work on ANE, even though it should based on my reading.
- do nothing
  - pros: as fast as it has been
  - pros: definitely works
  - cons: crawls if you have an initial prompt > 128 tokens (but you can trade off single token speed against larger initial prompts if you are commonly using large prompts -- maybe not bad).

Leaning towards giving enumerated inputs (just input_ids) a shot, initially ignoring qk_mask just to see if it runs on ANE. If not, probably not worth the time and can try option 1 (thinking it might not be terrible even though it seems a bit odd on paper).

Quick pass at enumerated inputs. Re-remembering why I avoid them. Even if only input_ids is varied, that translates to dynamic shapes as a result of concatenating the old and new KV caches. I'm guessing this is what causes it to not run on the ANE, but I could be misunderstanding. Maybe I need to open an issue.
-- Actually, I'm not so sure. I have pared away most things and it still won't run on ANE. It's basically: input_embeddings (1, 128, hidden_size) or (1,1,hidden_size) fed into a sequence of MLP+LN and only outputting a subset of logits. Runs 100% on ANE if the input is not enumerated. But cannot get it off CPU when the input is enumerated. Oh my. It seems that layer norm + enumerated shapes does not work with ANE. https://github.com/apple/coremltools/issues/1909 repros locally for me, but even removing LN from gpt2 doesn't fix it.

Would be very interesting to see if ml-ane-transformers layer norm suffers the same problem. Did some light hacking on the repro from 1909 and it seems like some 4D layer norm MIL ops do support ANE enumerated shapes, so there's hope.

Soo.. it might be possible to do the same for non-ANE optimized gpt2. Going to continue binary-searching the architecture for unsupported pieces. Realized it's easy to identify in a trace, since there's a purple ANE blob where it tries to load and an error in the os_log when it fails.
Chain of MLPs works, as long as there are no other inputs (interestingly).
x + mlp(x) chain works
~x = attn(x), x = x+mlp(x) chain works (note no layer norm pre-attn)~ edit: oops this was disconnected
| ^ plus a logits = torch.einsum('bid,jd->bij', x, splits[0]) does _not_ work
^ plus ln_f, as expected, does not work
^ plus ln_f but layer_norm MIL overrode to unsqueeze(0).layer_norm(-1).squeeze(0) _does_ work(!)
^ plus ln_2 in pre-mlp works
^ plus attn(x) does not work
^ plus attn(x) but only y = v and y.transpose(...) works (surprisingly that is noticeably slower)
^ plus q@k and @v (no softmax) (no kv cache at this point either) does not work (oof)
^ but without the scaling of q@k also does not work (long shot oh well)
^ tried replacing q@k with einsum -- does not work (but does save a transpose, might be interesting)

List of things that don't work:
layer_norm for rank 3 inputs
q@kT (maybe since the output has the same dynamic size in 2 dimensions? everywhere else is 1)

## October 26, 2023
Bad news/good news. I don't see a way to get the q@kT matrix multiplication working with enumerated shapes. Tried using the ml-ane-transformers einsum with no luck. However, according to Xcode performance tools, my iOS 17 device can do the matmul with enumerated shapes. So possible that upgrading OS is a solution. ... Switched to Sonoma, and yes, it also works on ANE.

Given that, think I will pause on the enumerated until I start doing dev on Sonoma. Will give the other full-size cache option a go, since even if I do get enumerated inputs working on Sonoma, those will be with full-size caches and the consistency will be nice. Plus feels good to make the model fully functional (esp. if enumerated fails for some other reason).

input_ids = (abcdefg)
context = 10, input length = 3
prompt batches = (xxa), (bcd), (efg) # x = pad
input cache => concat => multi-token, single-token
(xxxxxxx) => (xxxxxxxxxa) => (xxxxxxa), (xxxxxxx) # prompt 1, initialize input with pad
(xxxxxxa) => (xxxxxxabcd) => (xxxabcd), (xxxxxab) # prompt 2, use multi-token
(xxxabcd) => (xxxabcdefg) => (abcdefg), (xxabcde) # prompt 3, use multi-token
predicts 'h', next input is (fgh)
(xxabcde) => (xxabcdefgh) => (bcdefgh), (xabcdef) # generation 1, use single-token
predicts 'i', next input is (ghi)
(xabcdef) => (xabcdefghi) => (cdefghi), (abcdefg)
                remove leading 3 ^          ^ remove leading 1 + trailing 2

This is looking promising. Passing back two caches seems to add negligible (no?) overhead, and the generations from the single-token cache are accurate. Just need to run some tests to show that the prompt chunking approach actually works and we'll be home free.

## October 28, 2023
It works! Outputs match.

Thinking it might make sense to shrink the input length from 128. Recomputed this table (seems slightly slower than last time, maybe I have less battery, relative comparison is all that matters):

|model      |input length|median time (ms)|time for full prompt (ms)|time for 256 tokens (ms)|
|--         |            |                |                         |
|gpt2-large |128         |60              |240                      |15,360
|gpt2-large |96          |58.2            |349                      |14,899
|gpt2-large |64          |48.5            |388                      |12,416
|gpt2-large |32          |44              |704                      |11,264
|gpt2-large |2           |41              |10,496                   |10,496

time for full prompt = ceil(512 / input_length) * median time
time for 256 tokens = 256 * median time

64 seems like a sweet spot. A huge initial prompt is still acceptably fast, but we can generate slightly more tokens per second after that.

This is pretty cool. Basically a 4x speedup.

Moved the qk_mask building into the model. Had to rewrite it, but one less input to deal with is worth it. gpt2-large is the same speed too, so nothing sacrificed.