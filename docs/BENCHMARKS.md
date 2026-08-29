# Benchmarks

## Measured results - Orcarouter checkpoint only

Every runtime number here applies to the pinned
`orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` revision
`3a3b63161c0745390e5270179af42e46efc70799` on one DGX Spark. The MTP results
also use the exact 31 BF16 MTP tensors from pinned
`RadixArk/Qwen3.8-Flash-Next-NVFP4` revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`. Inferact and RadixArk are not
runtime-qualified by these results.

## Single-stream BF16 MTP depth sweep

The load guard verified 31/31 tensors before accepting requests. Each depth used a
32,768-token server profile, `concurrency=1`, one maximum sequence, 0.80 GPU-memory
utilization, one 128-token warm-up, then three fixed 256-token streamed samples.
The primary rate divides all completion tokens—including hidden reasoning—by total
request wall time, so it also includes latency before visible content. The next
column is the median time from request start to the first visible content delta.
Thinking mode used medium reasoning effort,
`temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`,
`presence_penalty=0.0`, and `repetition_penalty=1.0`.

| MTP depth | Median end-to-end completion tok/s | Time to first visible content | Aggregate acceptance |
| ---: | ---: | ---: | ---: |
| 0 | 27.26 | 1.196s | n/a |
| 1 | 33.17 | 1.206s | 305/463 (65.9%) |
| **2** | **44.23** | **1.324s** | **532/722 (73.7%)** |
| 3 | 40.95 | 1.629s | 563/993 (56.7%) |
| 4 | 41.84 | 1.123s | 611/1,148 (53.2%) |
| 5 | 35.45 | 1.249s | 454/1,560 (29.1%) |
| 6 | 35.71 | 1.429s | 516/1,230 (42.0%) |
| 8 | 30.43 | 1.758s | 487/1,600 (30.4%) |
| 10 | 28.31 | 1.648s | 642/2,150 (29.9%) |

`MTP=2` is the qualified selection. It was 62.2% faster end-to-end than `MTP=0`
in this single-stream short-request measurement. For MTP2, retained reasoning-token
accounting also gives a separate median visible-answer phase rate of 46.15 tok/s,
measured from the first through last visible content token. That rate excludes hidden
reasoning and must not be combined with total completion-token counts.

An earlier revision did combine all completion tokens with a visible-content-only
time interval. That mixed metric has been superseded and is not comparable to either
rate above. Higher draft depths lost enough acceptance to cost more verification
work than they saved. Depths five and above require the recipe's 48-token block
alignment for the QSA speculative ring.

The machine-readable record is
[`mtp-bf16-sweep.json`](../results/orcarouter/mtp-bf16-sweep.json).

## Native-context validation with MTP=2

| Measurement | Qualified observation |
| --- | --- |
| vLLM build | `0.1.dev20073+g8e685d198` |
| Model loading | 76.21 GiB |
| BF16 PLE on NVMe | 95.37 GiB |
| BF16 MTP overlay | 4.86 GiB; 31/31 tensors |
| BF16 KV available | 17.37 GiB |
| KV token capacity | 627,960 tokens |
| Native-window concurrency estimate | 2.40 at 262,144 tokens |
| Exact retrieval | pass at 240,051 prompt tokens |
| Retrieval TTFT / prefill | 116.80s / 2,055.17 prompt tok/s |

The full-profile model-load phase took 576.3 seconds. Qualified API readiness is
longer because runtime initialization follows weight loading; allow about 15 minutes
for a cold-start timeout rather than treating model-load time as readiness time.

## Baseline concurrency without the overlay

These older results use the original Orcarouter checkpoint at `MTP=0`, an 8K
request profile, and report whole-wave completion throughput rather than per-request
decode rates.

| Streams | Whole-wave completion throughput |
| ---: | ---: |
| 1 | 13.06 completion tok/s |
| 2 | 17.76 completion tok/s |
| 4 | 19.69 completion tok/s |
| 8 | 21.66 completion tok/s |

They are retained as baseline evidence. No claim is made that this is an MTP=2
concurrency sweep.

## Functional, vision, and stability evidence

The baseline qualification includes factual, code, JSON, tool, and vision checks.
The vision fixture contained three red squares and two blue circles and passed. The
MTP overlay additionally passed text generation, load-guard, acceptance, and exact
long-context retrieval checks.

The earlier file named `stability-2h.json` records only 904 seconds and is not a
completed two-hour run. Separately, `summary.json` records 8,159 seconds of
uninterrupted uptime at a gate, zero restarts, and eleven correct health probes.
These remain separate evidence and are not combined into a two-hour claim.
