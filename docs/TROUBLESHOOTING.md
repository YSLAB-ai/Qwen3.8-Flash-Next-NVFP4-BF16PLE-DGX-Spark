# Troubleshooting

## Gated checkpoint download fails

First sign in on the
[`orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` page](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4)
and accept the request to share your contact information. A CLI token cannot grant
access until the browser-side conditions have been accepted for that same account.

Then log the recipe into that account:

```bash
scripts/qwen38-dgx-spark login
```

Then rerun the recipe's download command. Do not bypass the checkpoint selection or
pinned revision checks.

## The disk gate rejects the download

The recipe requires at least 100 GiB free before it downloads a target. Free space on
the storage used for the checkpoint, preferably NVMe, then rerun the command. The PLE
is memory-mapped from that storage during serving.

## CUDA graph capture errors around PLE lookup

Use the recipe defaults. The lookup must be separated from captured graph segments,
so the runtime uses PIECEWISE capture. Do not switch this recipe to a full capture
mode for the mapped PLE path.

## KV cache configuration fails

Use the BF16 KV default. This model path requires BF16 KV, and an FP8 KV setting is
not a supported substitute.

## MTP appears enabled

Set `MTP=0`. The pinned Orcarouter checkpoint has zero MTP tensors. MTP1 provenance
showed 0/1,287 accepted only because an incomplete draft was constructed; it is
unsupported, not a tuning result. Depths 2-4 were not run for this checkpoint.

## Replacing a running recipe container

The serve command unloads an existing recipe container before it starts a replacement.
If the default Orcarouter attempt ended unexpectedly, run the exact guarded replacement:

```bash
scripts/qwen38-dgx-spark serve orca-uncensored --replace
```

For another manifest target, replace `orca-uncensored` with that target alias. Do not
start an additional container manually.

For supported target layouts, see [Compatibility](COMPATIBILITY.md). Benchmark scope
and limitations are in [Benchmarks](BENCHMARKS.md).
