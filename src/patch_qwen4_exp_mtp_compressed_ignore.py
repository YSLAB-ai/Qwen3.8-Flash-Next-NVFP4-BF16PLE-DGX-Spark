"""Remap compressed-tensors ignore paths for Qwen3.8's standalone MTP layer."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "qwen38-flash-dgx: remap compressed-tensors MTP ignore paths"
_METHOD_ANCHOR = "def _make_draft_vllm_config(\n"
_INSERTION_ANCHOR = "    draft_vllm_config = replace(\n"


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    for label, anchor in (
        ("Qwen3.8 draft-config method", _METHOD_ANCHOR),
        ("Qwen3.8 draft-config replacement", _INSERTION_ANCHOR),
    ):
        occurrences = source.count(anchor)
        if occurrences != 1:
            raise RuntimeError(f"expected exactly one {label} hook, found {occurrences}")

    block = f'''        # {MARKER}
        compressed_tensors_ignore = getattr(draft_quant_config, "ignore", None)
        if compressed_tensors_ignore:
            setattr(  # noqa: B010
                draft_quant_config,
                "ignore",
                _remap_ignored_layers(
                    compressed_tensors_ignore, mtp_start_layer_idx
                ),
            )

'''
    return source.replace(_INSERTION_ANCHOR, block + _INSERTION_ANCHOR, 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    args = parser.parse_args(argv)
    source = args.model_path.read_text(encoding="utf-8")
    patched = patch_source(source)
    if patched != source:
        args.model_path.write_text(patched, encoding="utf-8")
        print(f"patched compressed-tensors MTP ignore remap: {args.model_path}")
    else:
        print(f"compressed-tensors MTP ignore remap already present: {args.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
