# How it works

The recipe serves NVFP4 Qwen3.8-Flash-Next while retaining the checkpoint's
full-precision BF16 PLE on NVMe. A request needs selected lookup rows rather than the
entire table, so the patched runtime memory-maps the PLE safetensors data and gathers
those rows on demand.

## Memory layout

The qualified Orcarouter MTP profile loaded 76.21 GiB of model state. The complete
95.37 GiB BF16 PLE remains file-backed instead of being copied wholesale into DGX
Spark unified memory. At 0.80 GPU-memory utilization, the profile exposed 17.37 GiB
of BF16 KV, or 627,960 cached tokens.

## Runtime boundary

PLE gathering includes host work and a host-to-device transfer, so it must run
outside captured graph segments. The recipe uses PIECEWISE CUDA-graph capture with
the lookup declared as a splitting operation. BF16 KV is required by this path and
is not interchangeable with FP8 KV.

## BF16 MTP overlay

The Orcarouter repository contains no MTP weights. Its configuration alone cannot
produce a valid draft model. The `orca-uncensored-bf16-mtp` target obtains exactly
the 31 MTP tensors from a pinned, architecture-compatible RadixArk checkpoint,
verifies their filenames, byte sizes, hashes, names, and BF16 dtype, then writes a
4.86 GiB compact safetensors file into a deterministic local view.

The view links to the untouched Orcarouter checkpoint for all original tensors and
overlays only MTP tensors and the audited configuration needed to keep those native
modules unquantized. A runtime load guard requires 31/31 tensors; missing or
unexpected MTP state fails closed. This is a reproducible checkpoint view, not a
redistributed derivative model.

Depth testing selected `MTP=2`. At depths five and above, the recipe selects a
48-token attention block size so both the 12-token QSA speculative ring and the
kernel's 16-token alignment divide cleanly.

## Checkpoint-aware controls

The runtime validates each target's expected PLE layout before serving. Direct-BF16,
hybrid-BF16, and MTP-overlay modes have separate pinned identities and approved
filesystem roots. See [Compatibility](COMPATIBILITY.md); only the Orcarouter direct
and MTP-overlay paths have runtime qualification.

## Operational safeguards

Before download, the command enforces a 100 GiB free-disk gate. Gated Hugging Face
access is checked with a small file before the full transfer. Existing containers
can be replaced only when they carry this recipe's label, and the old container is
unloaded before the memory-capacity gate runs.

For measurements, see [Benchmarks](BENCHMARKS.md). For recovery guidance, see
[Troubleshooting](TROUBLESHOOTING.md).
