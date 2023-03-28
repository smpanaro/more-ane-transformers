# More ANE Transformers

Hardware-accelerated transformers on your Mac via CoreML. (Yes, that includes LLMs like GPT.)

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

**That model is tiny.** You can run larger models too:
|Model|Parameters|Size|Download|
|--|--|--|--|
|gpt2|124M|250MB|soon|
|gpt2-medium|350M|700MB|soon|
|gpt2-large|774M|1.5GB|soon|
|gpt2-xl|1558M|3GB|soon|

## Why CoreML?
Your Mac has custom hardware built for machine learning (✨the neural engine), but the only way to use it is through CoreML. Apple’s [ane-ml-transformers](https://github.com/apple/ml-ane-transformers) repo has patterns for making them go fast with the Neural Engine, but only one practical end-to-end example. This adds another.

## Is it fast?
The large (770M) model runs *~2 words/sec* running purely on Neural Engine + CPU. *~6 words/sec* if you have a new Mac with a fast GPU. Smaller models are faster, the xl (1.5B) is slower.

<img width="1077" alt="Xcode CoreML Performance test for gpt2-large" src="https://user-images.githubusercontent.com/2950214/228109875-84678093-4a96-4f99-9aa1-18d1d3340c25.png">

## What about iOS?
Smaller models (gpt2, maybe gpt2-medium) should run but larger models require too much memory. Quantization doesn’t help because CoreML uses float32 or float16 when running. 🤞 Apple changes this soon.

## Contribute
PRs welcome! New models ☑️ Fixing bugs ☑️ Speedups ☑️

## Thanks
This project really just stitches together previously open-sourced tools. Thanks y’all.
- [coremltools](https://github.com/apple/coremltools) - to make CoreML models
- [ane-ml-transformers](https://github.com/apple/ml-ane-transformers) - to make CoreML models go fast
- [nanoGPT](https://github.com/karpathy/nanoGPT) - for a hackable GPT2 implementation
- [huggingface](https://huggingface.co) - for weights + tokenizers
- [whisper_ane](https://github.com/Synopsis/whisper_ane) - for another ane example
- [Netron](https://netron.app) - for clutch visualization
- [ChatGPT](http://chat.openai.com) - for bouncing ideas
