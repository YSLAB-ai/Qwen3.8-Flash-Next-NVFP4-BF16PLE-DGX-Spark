#!/usr/bin/env bash
# Run the CPU-only PLE mmap checks in a previously built recipe image.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 IMAGE" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
docker run --rm --entrypoint python3 "$1" -c '
import ast
from pathlib import Path
path = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/mtp.py")
source = path.read_text(encoding="utf-8")
marker = "qwen38-flash-dgx: require exact native BF16 MTP checkpoint tensors"
assert marker in source, "native MTP load guard is absent"
ast.parse(source)
print("native MTP load guard present and syntax valid")
'
exec docker run --rm --entrypoint python3 \
  -v "$repo_root/src:/test:ro" \
  -w /test \
  "$1" test_ple_mmap_cpu.py
