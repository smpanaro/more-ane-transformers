# More ANE Transformers

Hardware-accelerated transformers on your Mac via CoreML. (Yes, that includes LLMs like GPT.)

> 🔌 Plug-n-play with preconverted CoreML models
> 
> 🔋 Performance with near-zero CPU usage
> 
> 🦍 Support for some of the largest Neural Engine models (up to 2.8B parameters)

## Try It
Generate text with a base gpt2 model like this:
```
❯ ./setup.sh && source env/bin/activate
❯ python3 convert.py
❯ python3 generate.py
Loading model from path gpt2.mlpackage using ComputeUnit.ALL...
Loaded model in 11.959s.

[Prompt] Before boarding your rocket to Mars, remember to pack these items on board:
...
```

That model is tiny—sometimes the results are a bit nonsensical. You can run larger models for better results:
|Model|Parameters|Size|Download|
|--|--|--|--|
|gpt2|124M|250MB|[link](https://github.com/smpanaro/more-ane-transformers/releases/tag/v0-2023-April-02)|
|gpt2-medium|350M|700MB|[link](https://github.com/smpanaro/more-ane-transformers/releases/tag/v0-2023-April-02)|
|gpt2-large|774M|1.5GB|[link](https://github.com/smpanaro/more-ane-transformers/releases/tag/v0-2023-April-02)|
|gpt2-xl|1558M|3GB|[link](https://github.com/smpanaro/more-ane-transformers/releases/tag/v0-2023-April-02)|
|pythia-1b|1011M|2GB|[link](https://github.com/smpanaro/more-ane-transformers/releases/tag/v0-2023-May-29)|

You can also see [evals/QUALITY.md](evals/QUALITY.md) for some example generations.

## Why CoreML?
Apple Silicon Macs have custom hardware built for machine learning (✨the neural engine). Its fast and power efficient but the only way to use it is through CoreML. This repo makes it easy.

## Is it fast?
The gpt2-xl (1.5B) model runs *~1.5 words/sec* running purely on Neural Engine. *~2.5 words/sec* if you have a new Mac with a fast GPU. Smaller models are faster (from a little to a lot -- every model is ~2x faster than the next largest).

See [evals/SPEED.md](evals/SPEED.md) for device benchmarks.

<img width="1074" alt="Xcode CoreML Performance test for gpt2-xl" src="https://user-images.githubusercontent.com/2950214/229385079-1ac5ee4c-3531-4e1d-bed3-cb870eee9158.png">
<sub>note this is prior to a 30% speedup (to 445ms), important part is how purple the bar is :)</sub>

## What about iOS?
Smaller models (gpt2, maybe gpt2-medium) should run but larger models require too much memory. Quantization doesn’t help because CoreML uses float32 or float16 when running. 🤞 Apple changes this soon.

## Can it run LLaMa?
No. Similar to iOS, this is mostly in Apple's hands as of May 2023. The smallest LLaMa model is too big at 7B params—M1 seems to have a hard limit at ~4GB. Even if it ran on the M2 neural engine, it would likely be slow (~2.5s/token assuming linear scaling).

## Contribute
PRs welcome! New models ☑️ Fixing bugs ☑️ Speedups ☑️

## Thanks
This project really just stitches together previously open-sourced tools. Thanks y’all.
- [coremltools](https://github.com/apple/coremltools) - to make CoreML models
- [ane-ml-transformers](https://github.com/apple/ml-ane-transformers) - to make CoreML models go fast
- [nanoGPT](https://github.com/karpathy/nanoGPT) - for a hackable GPT2 implementation
- [huggingface](https://huggingface.co) - for weights + tokenizers
- [ANE-Optimized-Whisper-OpenAI](https://github.com/RobertRiachi/ANE-Optimized-Whisper-OpenAI) - for splitting the embedding layer
- [whisper_ane](https://github.com/Synopsis/whisper_ane) - for another ane example
- [Netron](https://netron.app) - for clutch visualization
- [ChatGPT](http://chat.openai.com) - for bouncing ideas
