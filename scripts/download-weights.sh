#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/qwen38-dgx-spark" download "${1:-orca-uncensored}"
