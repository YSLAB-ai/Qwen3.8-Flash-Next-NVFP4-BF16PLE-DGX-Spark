"""Bridge the Transformers 5.16 Qwen4Exp layer-type spelling to this vLLM image."""

from __future__ import annotations

import argparse
from pathlib import Path


NEEDLE = "        super().__init__(layer_types=layer_types, **kwargs)\n"
MARKER = "qwen38-flash-dgx: normalize Transformers 5.16 QSA spelling"
REPLACEMENT = f'''\
        # {MARKER}
        if layer_types is not None:
            layer_types = [
                "full_attention"
                if layer_type == "qwen_sparse_attention"
                else layer_type
                for layer_type in layer_types
            ]
{NEEDLE}'''


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    occurrences = source.count(NEEDLE)
    if occurrences != 1:
        raise RuntimeError(
            "expected exactly one Qwen text-config constructor hook, "
            f"found {occurrences}"
        )
    return source.replace(NEEDLE, REPLACEMENT, 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path)
    args = parser.parse_args(argv)
    source = args.config_path.read_text(encoding="utf-8")
    patched = patch_source(source)
    if patched != source:
        args.config_path.write_text(patched, encoding="utf-8")
        print(f"patched Qwen4Exp config compatibility: {args.config_path}")
    else:
        print(f"Qwen4Exp config compatibility already present: {args.config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
