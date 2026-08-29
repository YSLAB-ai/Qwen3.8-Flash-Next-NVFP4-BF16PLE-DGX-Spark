# How it works

The recipe serves NVFP4 Qwen3.8-Flash-Next while retaining the checkpoint's
full-precision BF16 PLE on NVMe. The PLE is a lookup table: a request needs selected
rows, not the entire table at once. The patched runtime memory-maps the PLE
safetensors data and gathers those rows on demand.

## Memory layout

For the pinned Orcarouter checkpoint revision
`3a3b63161c0745390e5270179af42e46efc70799`, the measured model allocation was 71.13
GiB. The BF16 PLE itself is 95.37 GiB on disk and is not copied wholesale into DGX
Spark unified memory. This left approximately 20-21 GiB for the required BF16 KV
cache in the qualified run. NVMe matters because it backs the mapped PLE data.

## Runtime boundary

The PLE gather includes host work and a host-to-device transfer, so it must run
outside captured graph segments. The recipe therefore uses PIECEWISE CUDA-graph
capture with the lookup declared as a splitting operation. BF16 KV is required by
this model path; it is not interchangeable with an FP8 KV setting.

## Checkpoint-aware controls

The current pinned Orcarouter checkpoint has zero MTP tensors, despite advertising
one MTP layer in configuration. The recipe disables MTP (`MTP=0`) because an
incomplete draft is unsupported. The observed MTP1 0/1,287 acceptance was produced
by that incomplete draft and is invalid provenance, not a sampler outcome.

The runtime validates the selected target's expected PLE layout before serving. The
available direct-BF16 and hybrid-BF16 arrangements are described in
[Compatibility](COMPATIBILITY.md); only Orcarouter has runtime qualification.

## Operational safeguards

Before download, the command enforces a 100 GiB free-disk gate. It supports gated
Hugging Face login for checkpoints that require authorization. When a serve command
replaces an existing recipe container, it unloads the prior container first. These
guards keep the recipe's state and storage requirements explicit.

For measured behavior, including cold start and stability qualifications, see
[Benchmarks](BENCHMARKS.md). For setup failures and recovery, see
[Troubleshooting](TROUBLESHOOTING.md).
