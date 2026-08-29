# Qwen3.8-Flash-Next BF16 PLE recipe for DGX Spark

A recipe to run NVFP4 Qwen3.8-Flash-Next with its full-precision BF16 PLE memory-mapped from NVMe on a single NVIDIA DGX Spark (GB10).

## Mechanism

The Parameter Lookup Embedding (PLE) is a lookup table. This recipe leaves the
full-precision BF16 table in the checkpoint on NVMe and memory-maps it, gathering
only the rows needed by each request. That avoids placing the entire PLE in the
DGX Spark unified-memory pool while retaining the checkpoint's BF16 PLE values.
The runtime uses PIECEWISE CUDA-graph capture and a BF16 KV cache.

## Measured results - Orcarouter checkpoint only

All numbers below are for the pinned
[`orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4`](docs/COMPATIBILITY.md)
revision `3a3b63161c0745390e5270179af42e46efc70799` on one DGX Spark. They do
not describe Inferact or RadixArk.

- Model memory was 71.13 GiB, with a 95.37 GiB BF16 PLE memory-mapped from NVMe
  and approximately 20-21 GiB available for BF16 KV.
- Qualified cold readiness was 10m35s. Exact retrieval succeeded at 240,079 prompt
  tokens; that run measured 2,107.06 prompt tok/s and 27.39 decode tok/s.
- With MTP disabled (`MTP=0`), the short-request median was 28.57 tok/s.

See [benchmark provenance](docs/BENCHMARKS.md) for the method, concurrency,
vision, cold-start, and stability evidence.

## Compatibility matrix

| Checkpoint recipe | PLE arrangement | Runtime status |
| --- | --- | --- |
| Orcarouter | Direct BF16 PLE | runtime validated |
| Inferact | Direct BF16 PLE | runtime unvalidated |
| RadixArk | Hybrid BF16 PLE | runtime unvalidated |

The matrix is a tooling compatibility statement, not a runtime result. Details,
including pinned revisions, are in [Compatibility](docs/COMPATIBILITY.md).

## Requirements

- One NVIDIA DGX Spark (GB10) with its NVMe storage available to the recipe.
- At least 100 GiB free disk space before download; the command checks this gate.
- Access to gated Hugging Face checkpoint repositories and a logged-in Hugging Face
  client where the selected checkpoint requires it.
- Docker with NVIDIA runtime support.

## Quick start

```bash
git clone https://github.com/YSLAB-ai/Qwen3.8-Flash-Next-NVFP4-BF16PLE-DGX-Spark.git
cd Qwen3.8-Flash-Next-NVFP4-BF16PLE-DGX-Spark
docker build -t qwen38-flash-dgx .
scripts/qwen38-dgx-spark login
scripts/download-weights.sh
scripts/serve.sh
```

The default target is the pinned Orcarouter recipe. The wrapper validates the
checkpoint layout and disk gate before serving. Use `scripts/qwen38-dgx-spark
--help` for target and safety options.

## Safe defaults

The qualified Orcarouter default is native 262,144-token context, PIECEWISE capture,
BF16 KV, and `MTP=0`. The checkpoint advertises an MTP layer in its configuration but
contains zero MTP tensors. An MTP draft constructed without those tensors is
incomplete, so this recipe requires `MTP=0` and does not support MTP speculation for
this checkpoint. The runtime unloads a stopped or failed container before replacing
it, rather than running multiple model containers in the shared memory pool.

## Benchmark caveats

Measurements are point-in-time qualification evidence, not capacity guarantees.
The current stability file records 904 seconds elapsed, not a completed two-hour
run. Separate counters in the summary record 8,159 seconds of uninterrupted uptime
at a gate; they are separate evidence and must not be combined into a two-hour claim.

## Limitations

- The repository is a recipe only; it does not distribute model weights.
- Runtime qualification currently covers only the pinned Orcarouter checkpoint.
  Inferact and RadixArk are runtime unvalidated.
- The official Qwen checkpoint has a BF16 4B MTP head, but any future graft
  experiment is not validated and is not supported by this recipe.

## License

The code is licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for
attribution.

## Credits

- blazux originated the FP8 PLE mmap work.
- YSLAB-ai added BF16 PLE support, DGX Spark compatibility fixes,
  multi-checkpoint tooling, safeguards, and runtime qualification.
- Qwen and vLLM provide the model architecture and serving foundation.
