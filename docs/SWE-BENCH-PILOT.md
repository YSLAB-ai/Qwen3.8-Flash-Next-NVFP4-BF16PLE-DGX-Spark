# SWE-bench pilot

This is a small three-instance engineering pilot, not a leaderboard score. It was
run on one DGX Spark to compare the qualified Orcarouter BF16-PLE + BF16-MTP runtime
against an existing Unsloth UD-Q4_K_XL GGUF runtime of Qwen3.8-Flash-Next.

## Method

- Agent: [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent), one worker,
  30-step limit.
- Tasks: `astropy__astropy-12907`, `django__django-11099`, and
  `sympy__sympy-20590` from [SWE-bench](https://github.com/SWE-bench/SWE-bench).
- Sampler: thinking enabled, medium reasoning effort, `temperature=1.0`,
  `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, and
  `repetition_penalty=1.0`.
- MTP runtime: NVFP4 compute, full BF16 PLE mmap, BF16 MTP at depth two, vLLM
  `0.1.dev20073+g8e685d198`.
- GGUF runtime: Unsloth UD-Q4_K_XL, llama.cpp, concurrency one.
- The official task images are AMD64; evaluation used emulation on the DGX Spark's
  ARM64 host.

## Result

| Runtime | Generation wall time | Model calls | Prompt tokens | Completion tokens | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| NVFP4 + BF16 PLE + BF16 MTP2 | 6m44s | 24 | 92,864 | 4,943 | official evaluator: 3/3 resolved |
| UD-Q4_K_XL GGUF | 18m32s | 46 | 275,474 | 8,445 | identical patches to the 3/3 run |

All three generated patches were byte-for-byte identical between runtimes. The
MTP-backed predictions completed an official SWE-bench evaluation with three
resolved, zero unresolved, and zero infrastructure failures. A second official
evaluation of the identical GGUF patches was intentionally not repeated; exact
patch equality makes the task outcome identical.

This tiny sample indicates a large latency and agent-efficiency advantage for the
tested MTP runtime, but it cannot estimate general SWE-bench accuracy. A publishable
accuracy comparison would require a preregistered, much larger task set and native
evaluation capacity.
