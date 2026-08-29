# Compatibility

This repository has one runtime-qualified recipe and two additional checkpoint
layouts supported by the download and audit tooling. “Runtime unvalidated” means no
public claim is made that the target has completed this recipe's runtime
qualification.

| Target | Repository and pinned revision | PLE mode | Runtime status |
| --- | --- | --- | --- |
| Orcarouter | `orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` at `3a3b63161c0745390e5270179af42e46efc70799` | Direct BF16; optional BF16 MTP overlay | runtime validated |
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

## MTP overlay status

The pinned Orcarouter checkpoint contains zero MTP tensors even though its
configuration advertises an MTP layer. The separately named
`orca-uncensored-bf16-mtp` recipe overlays exactly 31 BF16 MTP tensors from pinned
`RadixArk/Qwen3.8-Flash-Next-NVFP4` revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`. Hash, dtype, name, architecture, and
load-count guards make this a checkpoint-specific operation. The overlay was
runtime validated with `MTP=2`; it is not claimed as a generic MTP source for the
other manifest targets.

See [Benchmarks](BENCHMARKS.md) for evidence restricted to Orcarouter and
[Troubleshooting](TROUBLESHOOTING.md) for disk, login, capture, KV, and unload
guidance.
