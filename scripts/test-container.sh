#!/usr/bin/env bash
# Run the CPU-only PLE mmap checks in a previously built recipe image.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 IMAGE" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
exec docker run --rm --entrypoint python3 \
  -v "$repo_root/src:/test:ro" \
  -w /test \
  "$1" test_ple_mmap_cpu.py
