# Simplest Setup

From the root of this repository run:
```shell
❯ ./setup.sh && source env/bin/activate
```

# Setup for Fastest Generation
You can speed up gpt2 text generation **2-5x** by installing a custom version of coremltools.

## Steps
1. If you have not already, complete the simple setup above.
1. Clone my fork of coremltools: `git clone https://github.com/smpanaro/coremltools.git` and run `git checkout more-ane-transformers`.
    1. You can browse the differences [here](https://github.com/apple/coremltools/compare/main...smpanaro:coremltools:more-ane-transformers) if you are cautious about installing forked repositories.
1. From this repo (more-ane-transformers), check the version of Python you are using.
    1. `python --version` from the command line will print it out.
1. Edit the coremltools Makefile on line 21 (`python = 3.7`) to specify your version.
    1. You just need the major and minor version (`3.10` not `3.10.12`).
1. From the coremltools repo, run `make wheel`. Wait for it to complete.
1. This will place a `.whl` file in the `build/dist/` directory of the coremltools repo.
1. From this repo (more-ane-transformers) install that.
    1. `pip install /path/to/coremltools/build/dist/coremltools-*.whl --force-reinstall`
1. Enjoy the speed!