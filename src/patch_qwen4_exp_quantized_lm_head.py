"""Give Qwen3.8 Flash-Next output heads their configured quantization method."""

from __future__ import annotations

import argparse
from pathlib import Path


NEEDLE = '''\
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
'''
MARKER = "qwen38-flash-dgx: support a quantized lm_head checkpoint"
REPLACEMENT = f'''\
            config.hidden_size,
            # {MARKER}
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
'''


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    occurrences = source.count(NEEDLE)
    if occurrences != 1:
        raise RuntimeError(
            "expected exactly one Qwen output-head constructor hook, "
            f"found {occurrences}"
        )
    return source.replace(NEEDLE, REPLACEMENT, 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_paths", type=Path, nargs="+")
    args = parser.parse_args(argv)
    for path in args.model_paths:
        source = path.read_text(encoding="utf-8")
        patched = patch_source(source)
        if patched != source:
            path.write_text(patched, encoding="utf-8")
            print(f"patched quantized Qwen output head: {path}")
        else:
            print(f"quantized Qwen output-head support already present: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
