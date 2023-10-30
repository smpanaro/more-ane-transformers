# More Neural Engine Transformers

Hardware-accelerated transformers on your Mac via CoreML. (Yes, that includes LLMs like GPT.)

> 🔋 Performance with near-zero CPU usage
>
> 🔌 Plug-n-play with preconverted CoreML models
>
> 🦍 Support for some of the largest Neural Engine models (up to 2.8B parameters)
>
> 🐍 Easy Python access to your Mac's hardware accelerators

## Try It
Generate text with a base gpt2 model like this:
```
❯ ./setup.sh && source env/bin/activate
❯ python3 convert.py
❯ python3 generate.py
Loading model from path gpt2.mlpackage using ComputeUnit.CPU_AND_NE...
Loaded model in 790.604ms.

[Prompt] Before boarding your rocket to Mars, remember to pack these items:
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
Apple Silicon Macs have custom hardware built for machine learning (✨the neural engine). Its fast and energy efficient but the only way to use it is through Apple's CoreML framework. This repo makes that easy.

## Is it fast?
The gpt2-xl model (1.5B) generates *~5 words/sec* (7.5 tokens/sec) running purely on Neural Engine. Smaller models are faster (every model is ~2x faster than the next largest).

See [evals/SPEED.md](evals/SPEED.md) for device benchmarks.

## What about iOS?
Smaller models (gpt2, gpt2-medium) should run on most devices. Depending on how much memory the device has larger models may also work. iOS 17 added support for runtime quantization which in theory will allow for larger models on all devices—none of the models in the repo use this yet.

## Can it run LLaMa?
Maybe. The smallest official LLaMa model is 4.5x the size of gpt2-xl. With runtime quantization (iOS17/macOS Sonoma+) and a newer device (M1 seems to have a model size limit of ~4GB) it might be possible.

## Contribute
PRs welcome! New models ☑️ Fixing bugs ☑️ Speedups ☑️

## Thanks
This project stitches together several previously open-sourced tools. Thanks y’all.
- [coremltools](https://github.com/apple/coremltools) - to make CoreML models
- [ane-ml-transformers](https://github.com/apple/ml-ane-transformers) - to make CoreML models go fast
- [nanoGPT](https://github.com/karpathy/nanoGPT) - for a hackable GPT2 implementation
- [huggingface](https://huggingface.co) - for weights + tokenizers
- [ANE-Optimized-Whisper-OpenAI](https://github.com/RobertRiachi/ANE-Optimized-Whisper-OpenAI) - for splitting the embedding layer
- [whisper.coreml](https://github.com/wangchou/whisper.coreml) - for an example of cross KV caches
- [whisper_ane](https://github.com/Synopsis/whisper_ane) - for another ane example
- [Netron](https://netron.app) - for clutch visualization
- [ChatGPT](http://chat.openai.com) - for bouncing ideas
