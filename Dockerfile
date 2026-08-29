# Qwen3.8-Flash-Next on a single DGX Spark / GB10, via vLLM.
#
# Starts from the official Qwen3.8-Flash-Next vLLM image and appends a narrow patch
# set. It serves the 51B-parameter n-gram ("PLE") table from disk via mmap instead of
# keeping it resident in the 128 GB unified pool. That is the single change that
# lets the ~176B (122 GiB NVFP4) checkpoint fit next to a real KV cache on one box.
#
#   docker build -t qwen38-flash-dgx .
#
# The base image is multi-arch (arm64 for the Spark's Grace CPU). Pinned by digest
# for reproducibility; bump the tag below if the upstream recipe moves.
FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

LABEL org.opencontainers.image.source="https://github.com/YSLAB-ai/Qwen3.8-Flash-Next-NVFP4-BF16PLE-DGX-Spark" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.base.digest="sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8" \
      ai.yslab.vllm.version="0.1.dev20073+g8e685d198"

# Package layout inside the official image (vLLM 0.1.dev20073, torch 2.13 cu130,
# numpy 2.2.6 — the patch needs numpy, already present).
ARG SP=/usr/local/lib/python3.12/dist-packages
ARG PLE=${SP}/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py
ARG QWEN_CONFIG=${SP}/vllm/models/qwen3_8_flash_next/config.py
ARG QWEN_MODEL=${SP}/vllm/models/qwen3_8_flash_next/nvidia/model.py
ARG QWEN_MTP=${SP}/vllm/models/qwen3_8_flash_next/nvidia/mtp.py
ARG LM_HEAD=${SP}/vllm/model_executor/layers/vocab_parallel_embedding.py

COPY src/vllm_ple_mmap.py ${SP}/vllm_ple_mmap.py
COPY src/patch_qwen4_exp_config.py /tmp/patch_qwen4_exp_config.py
COPY src/patch_qwen4_exp_quantized_lm_head.py /tmp/patch_qwen4_exp_quantized_lm_head.py
COPY src/patch_parallel_lm_head_linear_attrs.py /tmp/patch_parallel_lm_head_linear_attrs.py
COPY src/patch_qwen4_exp_mtp_load_guard.py /tmp/patch_qwen4_exp_mtp_load_guard.py
COPY src/patch_qwen4_exp_mtp_compressed_ignore.py /tmp/patch_qwen4_exp_mtp_compressed_ignore.py

# Append the hook to the model file. No-op unless VLLM_PLE_MMAP=1 at runtime, so
# the image still behaves exactly like upstream when the flag is off.
RUN python3 /tmp/patch_qwen4_exp_config.py ${QWEN_CONFIG} \
 && python3 /tmp/patch_qwen4_exp_quantized_lm_head.py ${QWEN_MODEL} ${QWEN_MTP} \
 && python3 /tmp/patch_parallel_lm_head_linear_attrs.py ${LM_HEAD} \
 && python3 /tmp/patch_qwen4_exp_mtp_compressed_ignore.py ${QWEN_MTP} \
 && python3 /tmp/patch_qwen4_exp_mtp_load_guard.py ${QWEN_MTP} \
 && cp ${PLE} ${PLE}.orig \
 && printf '\n\n# --- qwen38-flash-dgx: serve the PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)\n' >> ${PLE} \
 && python3 -c "import ast; [ast.parse(open(path).read()) for path in ('${PLE}', '${QWEN_CONFIG}', '${QWEN_MODEL}', '${QWEN_MTP}', '${LM_HEAD}')]; print('vLLM sources patched OK')" \
 && rm /tmp/patch_qwen4_exp_config.py /tmp/patch_qwen4_exp_quantized_lm_head.py /tmp/patch_parallel_lm_head_linear_attrs.py /tmp/patch_qwen4_exp_mtp_compressed_ignore.py /tmp/patch_qwen4_exp_mtp_load_guard.py
