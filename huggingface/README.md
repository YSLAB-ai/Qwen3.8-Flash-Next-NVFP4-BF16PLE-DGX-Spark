---
title: Qwen3.8-Flash-Next BF16 PLE DGX Spark recipe
library_name: vllm
tags:
  - qwen
  - vllm
  - dgx-spark
  - nvme
---

# Recipe only - no model weights

A recipe to run NVFP4 Qwen3.8-Flash-Next with its full-precision BF16 PLE memory-mapped from NVMe on a single NVIDIA DGX Spark (GB10).

This is an unposted Hugging Face card draft for the repository recipe. It contains no
model weights and does not redistribute any checkpoint.

The only runtime-validated target is the pinned Orcarouter checkpoint. Inferact and
RadixArk are runtime unvalidated. The Orcarouter recipe requires `MTP=0` because its
checkpoint has zero MTP tensors; an MTP graft experiment is not supported.

Read the repository [README](../README.md), [Compatibility](../docs/COMPATIBILITY.md),
and [benchmark provenance](../docs/BENCHMARKS.md) before use.
