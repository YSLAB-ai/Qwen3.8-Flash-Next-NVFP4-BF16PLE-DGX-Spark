# Compatibility

This repository has one runtime-qualified recipe and two additional checkpoint
layouts supported by the download and audit tooling. “Runtime unvalidated” means no
public claim is made that the target has completed this recipe's runtime
qualification.

| Target | Repository and pinned revision | PLE mode | Runtime status |
| --- | --- | --- | --- |
| Orcarouter | `orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` at `3a3b63161c0745390e5270179af42e46efc70799` | Direct BF16 | runtime validated |
| Inferact | configured in [`compatibility.json`](../compatibility.json) | Direct BF16 | runtime unvalidated |
| RadixArk | configured in [`compatibility.json`](../compatibility.json) | Hybrid BF16 | runtime unvalidated |

## Direct BF16

A direct-BF16 target carries its own complete BF16 PLE alongside the NVFP4 model
weights. The runtime validates the PLE tensor layout and memory-maps that table
directly from the selected checkpoint.

## Hybrid BF16

A hybrid-BF16 target uses NVFP4 model weights from its target repository and a
complete BF16 PLE from the separately pinned PLE source declared in
[`compatibility.json`](../compatibility.json). The tooling verifies both pinned
revisions and builds the combined view before serving. This is a compatibility
mechanism only: RadixArk remains runtime unvalidated.

## MTP status

The pinned Orcarouter checkpoint contains zero MTP tensors even though its
configuration advertises one MTP layer. Its validated recipe therefore requires
`MTP=0`. The official Qwen checkpoint has a BF16 4B MTP head, but a future graft
experiment has not been validated and is not supported.

See [Benchmarks](BENCHMARKS.md) for evidence restricted to Orcarouter and
[Troubleshooting](TROUBLESHOOTING.md) for disk, login, capture, KV, and unload
guidance.
