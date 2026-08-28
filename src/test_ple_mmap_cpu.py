"""CPU tests for dtype-aware PLE mmap gathering and checkpoint validation.

Run inside the vLLM image (needs numpy + torch, no GPU):
  docker run --rm -v $PWD:/t -w /t --entrypoint python3 vllm/vllm-openai:qwen38-flash-next test_ple_mmap_cpu.py
"""
import json
import os
import struct
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_ple_mmap as m  # noqa: E402

fp8 = m._resolve_ple_dtype("F8_E4M3")
assert fp8.torch_dtype == torch.float8_e4m3fn
assert fp8.itemsize == 1 and fp8.needs_scale is True
assert m._row_bytes(160, fp8) == 160

bf16 = m._resolve_ple_dtype("BF16")
assert bf16.torch_dtype == torch.bfloat16
assert bf16.itemsize == 2 and bf16.needs_scale is False
assert m._row_bytes(160, bf16) == 320

try:
    m._resolve_ple_dtype("F32")
    raise AssertionError("unsupported PLE dtype must fail")
except ValueError as exc:
    assert "unsupported PLE shard dtype" in str(exc)

assert m._read_required_scale("BF16", None) is None
try:
    m._read_required_scale("F8_E4M3", None)
    raise AssertionError("FP8 PLE without a scale must fail")
except RuntimeError as exc:
    assert "FP8 shards without ngram_embedding.weight_scale" in str(exc)

ROWS, COLS, PARTS = 100_000, 160, 8
shard_size = -(-ROWS // PARTS)
rng = np.random.default_rng(0)
table = rng.integers(0, 256, size=(ROWS, COLS), dtype=np.uint8)

tmp = tempfile.mkdtemp()
# write shards into 2 safetensors files (4 shards each) with a dummy tensor first,
# so data offsets are non-trivial
file_of = {}
for fi in range(2):
    tensors = {"dummy.weight": np.arange(37, dtype=np.float32).tobytes()}
    header = {"dummy.weight": {"dtype": "F32", "shape": [37], "data_offsets": [0, 37 * 4]}}
    off = 37 * 4
    for si in range(fi * 4, fi * 4 + 4):
        rows = table[si * shard_size : (si + 1) * shard_size]
        name = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{si}.weight"
        header[name] = {"dtype": "F8_E4M3", "shape": list(rows.shape), "data_offsets": [off, off + rows.nbytes]}
        tensors[name] = rows.tobytes()
        off += rows.nbytes
        file_of[name] = f"model-plefp8-0000{fi}.safetensors"
    if fi == 1:
        name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
        header[name] = {"dtype": "F32", "shape": [], "data_offsets": [off, off + 4]}
        tensors[name] = struct.pack("<f", 0.03125)
        off += 4
        file_of[name] = f"model-plefp8-0000{fi}.safetensors"
    hb = json.dumps(header).encode()
    with open(os.path.join(tmp, f"model-plefp8-0000{fi}.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for name in header:
            f.write(tensors[name])
with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as f:
    json.dump({"weight_map": file_of}, f)

shards, dtype_str, scale_entry = m._find_shards(tmp, 1)
cols = shards.pop("__cols__")
assert dtype_str == "F8_E4M3" and cols == COLS, (dtype_str, cols)
assert len(shards) == PARTS, len(shards)
assert abs(float(m._read_scale(scale_entry)) - 0.03125) < 1e-9
for idx, (_p, _o, rows) in shards.items():
    assert rows == max(0, min(shard_size, ROWS - idx * shard_size)), (idx, rows)

t = m.MmapPleTable(shards, shard_size, cols, torch.float8_e4m3fn, workers=8, chunk=512)
assert t.rows_total == ROWS

for n in (1, 16, 5000, 131_072):
    ids = rng.integers(0, ROWS, size=n, dtype=np.int64)
    ids[: n // 3] = ids[0]  # lots of duplicates, like real n-grams
    t0 = time.perf_counter()
    got = t.gather(ids)
    dt = time.perf_counter() - t0
    ref = table[ids]
    assert got.shape == (n, COLS) and got.dtype == np.uint8
    assert np.array_equal(got, ref), f"mismatch for n={n}"
    print(f"gather n={n:>7}: OK in {dt*1e3:7.2f} ms")

# torch view path used by the placeholder
emb = m._MmapNgramEmbedding(ROWS, COLS)
emb.table = t
ids_t = torch.from_numpy(rng.integers(0, ROWS, size=(300, 16), dtype=np.int64))
out = emb(ids_t)
assert out.shape == (300, 16, COLS) and out.dtype == torch.float8_e4m3fn
assert np.array_equal(out.view(torch.uint8).numpy().reshape(-1, COLS), table[ids_t.numpy().reshape(-1)])
print("placeholder forward: OK (fp8 view, shape", tuple(out.shape), ")")

# BF16 checkpoint path: exact values, a partial final shard, cross-shard ids,
# duplicates, and a non-trivial safetensors data offset.
BF_ROWS, BF_COLS, BF_PARTS = 1_031, 7, 8
bf_shard_size = -(-BF_ROWS // BF_PARTS)
bf_ref = (
    torch.arange(BF_ROWS * BF_COLS, dtype=torch.float32)
    .remainder(997)
    .div(31)
    .to(torch.bfloat16)
    .reshape(BF_ROWS, BF_COLS)
)
bf_u8 = bf_ref.view(torch.uint8).numpy().reshape(BF_ROWS, BF_COLS * 2)
bf_tmp = tempfile.mkdtemp()
bf_file_of = {}
for fi in range(2):
    tensors = {"dummy.weight": np.arange(11, dtype=np.float32).tobytes()}
    header = {
        "dummy.weight": {
            "dtype": "F32",
            "shape": [11],
            "data_offsets": [0, 11 * 4],
        }
    }
    off = 11 * 4
    for si in range(fi * 4, fi * 4 + 4):
        rows = bf_u8[si * bf_shard_size : (si + 1) * bf_shard_size]
        name = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{si}.weight"
        header[name] = {
            "dtype": "BF16",
            "shape": [len(rows), BF_COLS],
            "data_offsets": [off, off + rows.nbytes],
        }
        tensors[name] = rows.tobytes()
        off += rows.nbytes
        bf_file_of[name] = f"model-plebf16-{fi}.safetensors"
    hb = json.dumps(header).encode()
    with open(os.path.join(bf_tmp, f"model-plebf16-{fi}.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for name in header:
            f.write(tensors[name])
with open(os.path.join(bf_tmp, "model.safetensors.index.json"), "w") as f:
    json.dump({"weight_map": bf_file_of}, f)

bf_shards, bf_dtype, bf_scale = m._find_shards(bf_tmp, 1)
bf_cols = bf_shards.pop("__cols__")
assert bf_dtype == "BF16" and bf_scale is None and bf_cols == BF_COLS
bf_table = m._open_ple_table(
    bf_shards,
    bf_shard_size,
    bf_cols,
    bf_dtype,
    workers=4,
    chunk=32,
)
bf_ids_np = np.array(
    [0, bf_shard_size - 1, bf_shard_size, BF_ROWS - 1, 3, 3],
    dtype=np.int64,
)
bf_emb = m._MmapNgramEmbedding(BF_ROWS, BF_COLS)
bf_emb.table = bf_table
bf_got = bf_emb(torch.from_numpy(bf_ids_np))
assert bf_got.dtype == torch.bfloat16
assert torch.equal(bf_got.cpu(), bf_ref[torch.from_numpy(bf_ids_np)])
assert bf_table.row_bytes == BF_COLS * 2
print("placeholder forward: OK (bf16 exact values, shape", tuple(bf_got.shape), ")")

width_tmp = tempfile.mkdtemp()
width_names = [
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight",
]
width_header = {
    width_names[0]: {"dtype": "BF16", "shape": [1, 7], "data_offsets": [0, 14]},
    width_names[1]: {"dtype": "BF16", "shape": [1, 8], "data_offsets": [14, 30]},
}
width_hb = json.dumps(width_header).encode()
width_file = os.path.join(width_tmp, "model-width.safetensors")
with open(width_file, "wb") as f:
    f.write(struct.pack("<Q", len(width_hb)))
    f.write(width_hb)
    f.write(bytes(30))
with open(os.path.join(width_tmp, "model.safetensors.index.json"), "w") as f:
    json.dump({"weight_map": {name: "model-width.safetensors" for name in width_names}}, f)
try:
    m._find_shards(width_tmp, 1)
    raise AssertionError("mixed PLE shard widths must fail")
except ValueError as exc:
    assert "mixed widths" in str(exc)

layout_shards = {
    0: ("a", 0, 9),
    1: ("b", 0, 9),
    2: ("c", 0, 7),
}
assert m._validate_shard_layout(layout_shards, parts=3, vocab=25) == 9

try:
    m._validate_shard_layout({0: layout_shards[0], 2: layout_shards[2]}, parts=3, vocab=25)
    raise AssertionError("a missing middle PLE shard must fail")
except RuntimeError as exc:
    assert "PLE shard indices" in str(exc) and "[0, 2]" in str(exc)

bad_rows = dict(layout_shards)
bad_rows[2] = ("c", 0, 6)
try:
    m._validate_shard_layout(bad_rows, parts=3, vocab=25)
    raise AssertionError("a PLE shard with the wrong row count must fail")
except RuntimeError as exc:
    assert "PLE shard 2 has 6 rows, expected 7" in str(exc)

# zeros path (no table)
emb2 = m._MmapNgramEmbedding(ROWS, COLS)
z = emb2(ids_t)
assert z.shape == (300, 16, COLS) and float(z.abs().sum()) == 0.0
print("zeros path: OK")

# out-of-range must raise, not corrupt
try:
    t.gather(np.array([ROWS + 5], dtype=np.int64))
    raise SystemExit("expected IndexError")
except IndexError:
    print("out-of-range: raises IndexError OK")

t.prewarm()
print("prewarm: OK")
print("ALL OK")
