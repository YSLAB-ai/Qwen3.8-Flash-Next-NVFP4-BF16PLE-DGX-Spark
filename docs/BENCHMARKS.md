# Benchmarks

## Measured results - Orcarouter checkpoint only

Every number in this document is evidence for the pinned
`orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` revision
`3a3b63161c0745390e5270179af42e46efc70799` on a single DGX Spark. Inferact and
RadixArk are runtime unvalidated; none of these results transfers to them.

| Measurement | Qualified observation |
| --- | --- |
| Model memory | 71.13 GiB |
| BF16 PLE on NVMe | 95.37 GiB |
| BF16 KV | approximately 20-21 GiB |
| Cold readiness | 10m35s qualified cold readiness |
| Native retrieval | exact retrieval at 240,079 prompt tokens |
| Retrieval throughput | 2,107.06 prompt tok/s and 27.39 decode tok/s on that run |

## MTP provenance

| Configuration | Observation | Interpretation |
| --- | --- | --- |
| MTP0 | 28.57 tok/s short median | Qualified default; MTP is disabled. |
| MTP1 | 21.09 tok/s median with 0/1,287 accepted | Unsupported: the checkpoint lacks MTP tensors, so vLLM constructed an incomplete draft. This is not a sampler result. |
| MTP depths 2-4 | Not run | Not run for this checkpoint. |

The checkpoint advertises one MTP layer in configuration but has zero MTP tensors.
`MTP=0` is mandatory for the pinned recipe. The official Qwen checkpoint's BF16 4B
MTP head does not make a future graft experiment supported or validated.

In the exact evidence wording, the qualified result is **28.57 tok/s MTP0 short
median**. The recorded **21.09 tok/s MTP1 median with 0/1,287 accepted** belongs to
the incomplete, unsupported draft and is not a sampler result.

## Concurrency

These are whole-wave completion throughput values, not per-request decode rates.

| Streams | Whole-wave completion throughput |
| ---: | ---: |
| 1 | 13.06 completion tok/s |
| 2 | 17.76 completion tok/s |
| 4 | 19.69 completion tok/s |
| 8 | 21.66 completion tok/s |

Equivalently, whole-wave concurrency throughput was 13.06, 17.76, 19.69, and 21.66
completion tok/s for 1, 2, 4, and 8 streams respectively.

## Functional and vision checks

The qualified evidence includes factual, code, JSON, tool, and vision checks. The
vision fixture contained three red squares and two blue circles, and the recorded
vision result passed. These checks establish only the tested Orcarouter run.

## Cold start and stability evidence

Cold readiness was qualified at 10m35s. The file named `stability-2h.json` records
904 seconds elapsed, so it is not a completed two-hour run. Separately,
`summary.json` records 8,159 seconds of uninterrupted uptime at a gate, zero
container restarts, and eleven correct health probes. The files are separate evidence
and must not be combined into a two-hour stability claim.

The sanitized records are in [`results/orcarouter`](../results/orcarouter). For
runtime assumptions, see [How it works](HOW-IT-WORKS.md).
