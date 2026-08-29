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

## MTP load guard reports fewer than 31/31 tensors

Do not bypass the guard. Re-run preparation for the overlay so the recipe verifies
and rebuilds its compact BF16 MTP file from the pinned source shards:

```bash
scripts/qwen38-dgx-spark prepare orca-uncensored-bf16-mtp
```

The original `orca-uncensored` target intentionally has no MTP weights. Use the
separately named overlay target when passing `--mtp`.

## Deep MTP reports a QSA capacity or block-size error

Use the recipe command rather than constructing a raw vLLM invocation. The recipe
automatically selects the required 48-token block alignment for depths five and
higher. The qualified setting remains `MTP=2`.

## Replacing a running recipe container

The serve command unloads an existing recipe container before it starts a replacement.
If the qualified Orcarouter MTP attempt ended unexpectedly, run the guarded replacement:

```bash
scripts/qwen38-dgx-spark serve orca-uncensored-bf16-mtp --mtp 2 --replace
```

For another manifest target, replace the target alias and use only options supported
for that target. Do not start an additional model container manually.

For supported target layouts, see [Compatibility](COMPATIBILITY.md). Benchmark scope
and limitations are in [Benchmarks](BENCHMARKS.md).
