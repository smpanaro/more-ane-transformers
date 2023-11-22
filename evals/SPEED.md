# Neural Engine Benchmarks

Average times to generate a token as reported by `python generate.py --model_path $MODEL_PATH --compute_unit CPUAndANE`. All models have an input length of 512. All run almost entirely (~99%) on the Neural Engine except where noted.

NOTE: These times are intended to give a rough relative measure of performance and are not overly-rigorous.

|Legend||
|-|-|
|×| model too big for neural engine|
|-| not tested|
|¦| model was split into pieces and merged via a pipeline in order to run on ANE|

## gpt2

All gpt2 timings use a performance-optimized (but equally accurate) version of coremltools. See [SETUP.md](SETUP.md) for details.

|Device     |gpt2 (124M)|gpt2-medium (350M)|gpt2-large (774M)|gpt2-xl (1558M)¦|
|-|-|-|-|-|
|2021 MBP M1|26ms*      |27ms              |51ms             |138ms           |
|2022 Air M2|21ms*      |23ms              |46ms             |95ms            |

\* partially runs on CPU

## pythia

|Device     |70M  |160M|410M |1B   |1.4B¦ |2.8B¦ |6.9B|
|-|-|-|-|-|-|-|-|
|2021 MBP M1|16ms*|39ms|112ms|304ms|465ms |×     |×   |
|2022 Air M2|-    |-   |-    |-    |-     |1050ms|×†  |

\* partially runs on CPU
† may still be possible
