# Qwen3.8-Flash-Next BF16 PLE recipe for DGX Spark

A recipe to run NVFP4 Qwen3.8-Flash-Next with its full-precision BF16 PLE memory-mapped from NVMe on a single NVIDIA DGX Spark (GB10).

## What this recipe does

The Parameter Lookup Embedding (PLE) is a lookup table. This recipe keeps the
complete 95.37 GiB BF16 PLE in its checkpoint on NVMe and memory-maps it, gathering
only the rows needed by each request. The NVFP4 compute trunk stays in unified
memory while the full table remains file-backed. The runtime uses PIECEWISE CUDA
graphs and a BF16 KV cache.

For the qualified Orcarouter target, the recipe can also build a deterministic
overlay containing the 31 native BF16 MTP tensors from a pinned RadixArk checkpoint.
The original gated Orcarouter files are not modified or redistributed.

## Measured results - Orcarouter checkpoint only

These results apply only to
[`orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4`](docs/COMPATIBILITY.md) revision
`3a3b63161c0745390e5270179af42e46efc70799` plus the documented BF16 MTP overlay,
on one DGX Spark. They do not describe Inferact or RadixArk runtime performance.

- The MTP load guard verified 31/31 expected tensors. The compact BF16 MTP file is
  4.86 GiB.
- A single-stream MTP depth sweep selected `MTP=2`: 44.23 tok/s end-to-end,
  including hidden reasoning and first-visible latency, versus 27.26 tok/s
  at `MTP=0`, a 62.2% increase. Its separately measured visible-answer phase was
  46.15 tok/s, with 1.324s to first visible content.
- Native 262,144-token startup used 76.21 GiB for model loading and exposed 17.37
  GiB of BF16 KV, enough for 627,960 cached tokens (2.40 concurrent native windows).
- Exact retrieval passed at 240,051 prompt tokens, with 116.80s TTFT and 2,055.17
  prompt tok/s.

See [Benchmarks](docs/BENCHMARKS.md) for the full MTP sweep, sampling parameters,
baseline concurrency results, and caveats. A separate
[three-instance SWE-bench pilot](docs/SWE-BENCH-PILOT.md) records an agent-level
comparison against the installed UD-Q4_K_XL GGUF runtime.

## Compatibility matrix

| Checkpoint recipe | PLE arrangement | Runtime status |
| --- | --- | --- |
| Orcarouter | Direct BF16 PLE; optional BF16 MTP overlay | runtime validated |
| Inferact | Direct BF16 PLE | runtime unvalidated |
| RadixArk | Hybrid BF16 PLE | runtime unvalidated |

The matrix describes tooling compatibility, not equivalent model behavior. See
[Compatibility](docs/COMPATIBILITY.md) for pinned revisions and boundaries.

## Requirements

- One NVIDIA DGX Spark (GB10), with NVMe storage available to the recipe.
- At least 100 GiB free disk space before download; the command checks this gate.
- A Hugging Face account that has accepted the Orcarouter checkpoint's access
  conditions. Open the
  [`orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` page](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4),
  sign in, and accept the request to share your contact information. A read token
  cannot grant gated access until this browser step is complete.
- Docker with NVIDIA runtime support.

## Quick start: qualified BF16 MTP profile

```bash
git clone https://github.com/YSLAB-ai/Qwen3.8-Flash-Next-NVFP4-BF16PLE-DGX-Spark.git
cd Qwen3.8-Flash-Next-NVFP4-BF16PLE-DGX-Spark
docker build -t qwen38-flash-dgx .
scripts/qwen38-dgx-spark login
scripts/qwen38-dgx-spark prepare orca-uncensored-bf16-mtp
scripts/qwen38-dgx-spark serve orca-uncensored-bf16-mtp --mtp 2
```

The interactive login stores the token in the same recipe cache used by the
downloader. Before the large transfer, the downloader requests only `config.json`
so gated-access problems fail early. Preparation then downloads the pinned
Orcarouter checkpoint, only the three pinned source shards containing the 31 MTP
tensors, and writes an audited local overlay.

The qualified profile is loopback-only, native 262,144-token context, eight maximum
sequences, PIECEWISE capture, BF16 KV, no PLE prewarm, and `MTP=2`. The runtime
unloads an existing recipe-labelled container before a guarded replacement.

## Recommended thinking sampler

The MTP sweep used thinking mode with medium reasoning effort and:

```text
temperature=1.0 top_p=0.95 top_k=20 min_p=0.0
presence_penalty=0.0 repetition_penalty=1.0
```

Reasoning effort is a request/client setting; it is not baked into the weights.

## Limitations

- This repository is a recipe only and does not distribute model weights.
- Runtime qualification covers only the pinned Orcarouter checkpoint and its
  documented RadixArk-sourced BF16 MTP overlay. Inferact and RadixArk remain runtime
  unvalidated.
- The overlay assumes matching Qwen3.8-Flash-Next architecture and exact pinned
  tensor identities. It is audited rather than treated as a generic plug-in.
- Measurements are point-in-time observations, not capacity guarantees.

## License

The recipe code is licensed under [Apache-2.0](LICENSE). Model use remains subject
to the licenses and access terms of the referenced repositories. See
[NOTICE](NOTICE) for attribution.

## Credits

- blazux originated the FP8 PLE mmap work.
- YSLAB-ai added BF16 PLE support, DGX Spark compatibility fixes, checkpoint-aware
  preparation, the guarded BF16 MTP overlay, and runtime qualification.
- Qwen and vLLM provide the model architecture and serving foundation.
