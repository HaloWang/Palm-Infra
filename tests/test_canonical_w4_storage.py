#!/usr/bin/env python3
"""Validate the one canonical package layout for each supported W4 group."""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "models"))

from transpile import (  # noqa: E402
    _read_weight_header,
    _validate_package_weight_header,
)


HEADER = struct.Struct("<IIIIQQQQQQQQII")
MAGIC = 0x50414D58
PRECISION_INT4 = 3
FLAG_BG128 = 1 << 1
FLAG_BG32 = 1 << 2


def quantize(executable: str, root: Path, rows: int, cols: int,
             group_size: int) -> tuple[Path, tuple[int, ...]]:
    source = root / f"w4g{group_size}.f32"
    output = root / f"w4g{group_size}.weights"
    values = [((i * 17) % 29 - 14) * 0.03125
              for i in range(rows * cols)]
    source.write_bytes(struct.pack(f"<{len(values)}f", *values))
    subprocess.run(
        [executable, str(source), str(output), "f32", str(rows),
         str(cols), "w4", str(group_size), "1"],
        check=True)
    raw = output.read_bytes()
    return output, HEADER.unpack(raw[:HEADER.size])


def check_layout(executable: str, root: Path, rows: int, cols: int,
                 group_size: int, flag: int, block_bytes: int):
    output, header = quantize(executable, root, rows, cols, group_size)
    (magic, flags, _, precision, n, k, _, _, data_offset, data_size,
     scales_offset, scales_size, stored_group, num_groups) = header
    groups_per_row = cols // group_size
    expected_data = ((rows + 7) // 8) * groups_per_row * block_bytes
    assert magic == MAGIC
    assert flags == flag
    assert precision == PRECISION_INT4
    assert (n, k) == (rows, cols)
    assert data_offset == HEADER.size
    assert data_size == expected_data
    assert scales_offset == 0 and scales_size == 0
    assert stored_group == group_size
    assert num_groups == rows * groups_per_row
    assert output.stat().st_size == HEADER.size + expected_data
    _validate_package_weight_header(
        output.name, _read_weight_header(str(output)))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: test_canonical_w4_storage.py <mollm-quantize>",
              file=sys.stderr)
        return 2
    executable = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="mollm_w4_storage_") as tmp:
        root = Path(tmp)
        check_layout(executable, root, 9, 64, 32, FLAG_BG32, 160)
        check_layout(executable, root, 9, 128, 128, FLAG_BG128, 544)

        source = root / "unsupported.f32"
        source.write_bytes(bytes(8 * 64 * 4))
        result = subprocess.run(
            [executable, str(source), str(root / "unsupported.weights"),
             "f32", "8", "64", "w4", "64", "1"],
            capture_output=True, text=True)
        assert result.returncode != 0
        assert "group_size=32 or 128" in result.stderr

        legacy = {
            "precision": PRECISION_INT4,
            "flags": 1,
            "shape": [8, 32, 1, 1],
            "scales_size": 8 * 4,
            "group_size": 32,
            "num_groups": 8,
            "data_size": 8 * 32 // 2,
        }
        try:
            _validate_package_weight_header("legacy.weights", legacy)
        except ValueError as error:
            assert "legacy Q4DOT" in str(error)
        else:
            raise AssertionError("legacy Q4DOT package storage was accepted")

    print("Canonical W4 storage tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
