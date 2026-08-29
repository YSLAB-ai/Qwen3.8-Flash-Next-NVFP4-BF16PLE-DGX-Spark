#!/usr/bin/env bash
# Guarded wrapper for the qualified BF16 PLE runtime profile.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/qwen38-dgx-spark" serve "${1:-orca-uncensored}" "${@:2}"
