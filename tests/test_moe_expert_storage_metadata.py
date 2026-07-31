#!/usr/bin/env python3
"""Smoke tests for MoE expert storage package metadata."""

import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from transpile import (
    GraphBuilder,
    Precision,
    StreamedWeight,
    WEIGHT_FLAG_INT4_BG32,
    WEIGHT_FLAG_FP8_BLOCK128,
    WeightByteRange,
    _write_weight_file,
    save_package,
)


def read_package_metadata(path: str) -> dict:
    with open(path, "rb") as f:
        header = f.read(128)
        meta_off, meta_len = struct.unpack_from("<QQ", header, 8)
        f.seek(meta_off)
        return json.loads(f.read(meta_len))


def read_package_weights_offset(path: str) -> int:
    with open(path, "rb") as f:
        return struct.unpack_from("<Q", f.read(128), 88)[0]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        weights_dir = Path(tmp) / "weights"
        weights_dir.mkdir()

        gate_up_name = "layer0_experts_gate_up.weights"
        down_name = "layer0_experts_down.weights"

        # Logical gate_up: E=2, rows_per_expert=8, K=32. Canonical BG32
        # stores one self-contained 160-byte block per expert.
        gate_up_q4 = np.arange(2 * 160, dtype=np.uint8)
        _write_weight_file(
            str(weights_dir / gate_up_name),
            gate_up_q4,
            group_size=32,
            num_groups=16,
            precision=Precision.INT4,
            logical_shape=(16, 32),
            flags=WEIGHT_FLAG_INT4_BG32,
        )

        # Logical down uses the same canonical blocks and is streamed in
        # expert-major order without a separate scale plane.
        down_q4 = np.arange(2 * 160, dtype=np.uint8)
        down_source = Path(tmp) / "down-source.bin"
        down_source.write_bytes(down_q4.tobytes())
        streamed_down = StreamedWeight(
            precision=Precision.INT4,
            logical_shape=(16, 32),
            data_ranges=[
                WeightByteRange(str(down_source), 0, down_q4.nbytes),
            ],
            group_size=32,
            num_groups=16,
            flags=WEIGHT_FLAG_INT4_BG32,
            expert_interleave_count=2,
        )

        g = GraphBuilder()
        x = g.input("x", (32, 1), prec=Precision.FP32)
        gate_up = g.weight("./" + gate_up_name, (16, 32), Precision.INT4)
        down = g.weight("./" + down_name, (16, 32), Precision.INT4)
        g.matmul(x, gate_up)
        g.matmul(x, down)

        package_path = str(Path(tmp) / "test.mollm")
        save_package(
            package_path,
            g,
            g,
            str(weights_dir),
            {
                "num_experts": 2,
                "moe_expert_storage": {
                    "version": 1,
                    "layout": "aggregate_rows_v1",
                    "num_experts": 2,
                    "layers": [{
                        "layer": 0,
                        "num_experts": 2,
                        "gate_up": {
                            "weight": "./" + gate_up_name,
                            "rows_per_expert": 8,
                            "cols": 32,
                        },
                        "down": {
                            "weight": "./" + down_name,
                            "rows_per_expert": 8,
                            "cols": 32,
                        },
                    }],
                },
            },
            streamed_weights={"./" + down_name: streamed_down},
        )

        metadata = read_package_metadata(package_path)
        layer = metadata["moe_expert_storage"]["layers"][0]
        gate_up_meta = layer["gate_up"]
        down_meta = layer["down"]

        assert gate_up_meta["precision"] == int(Precision.INT4)
        assert gate_up_meta["shape"][:2] == [16, 32]
        assert gate_up_meta["data_offset"] == 88
        assert gate_up_meta["scales_offset"] == 0
        assert gate_up_meta["group_size"] == 32
        assert gate_up_meta["groups_per_row"] == 1
        assert gate_up_meta["expert_data_bytes"] == gate_up_q4.nbytes // 2
        assert gate_up_meta["expert_scales_bytes"] == 0

        assert down_meta["shape"][:2] == [16, 32]
        assert down_meta["expert_data_bytes"] == down_q4.nbytes // 2
        assert down_meta["expert_scales_bytes"] == 0
        assert "expert_stride" not in down_meta
        assert down_meta["data_offset"] == 88
        assert down_meta["scales_offset"] == 0
        assert down_meta["weight_offset"] > gate_up_meta["weight_offset"]

        weights_offset = read_package_weights_offset(package_path)
        with open(package_path, "rb") as package:
            package.seek(
                weights_offset + down_meta["weight_offset"] + 88)
            payload = package.read(down_q4.nbytes)
        data_per_expert = down_q4.nbytes // 2
        expected = bytearray()
        for expert in range(2):
            expected += down_q4.tobytes()[
                expert * data_per_expert:(expert + 1) * data_per_expert]
        assert payload == expected

        fp8_name = "layer0_experts_fp8.weights"
        fp8_values = np.full((256, 128), 0x38, dtype=np.uint8)
        fp8_scales = np.array([127, 126], dtype=np.uint8)
        _write_weight_file(
            str(weights_dir / fp8_name), fp8_values,
            scales=fp8_scales, group_size=128,
            num_groups=fp8_scales.size,
            precision=Precision.FP8_E4M3,
            logical_shape=fp8_values.shape,
            flags=WEIGHT_FLAG_FP8_BLOCK128,
            scale_dtype=np.uint8,
        )
        fp8_graph = GraphBuilder()
        fp8_x = fp8_graph.input("x", (128, 1), prec=Precision.FP32)
        fp8_weight = fp8_graph.weight(
            "./" + fp8_name, (256, 128), Precision.FP8_E4M3)
        fp8_graph.matmul(fp8_x, fp8_weight)

        def fp8_metadata(rows_per_expert: int,
                         num_experts: int = 2) -> dict:
            return {
                "num_experts": num_experts,
                "moe_expert_storage": {
                    "version": 1,
                    "layout": "aggregate_rows_v1",
                    "num_experts": num_experts,
                    "layers": [{
                        "layer": 0,
                        "num_experts": num_experts,
                        "gate_up": {
                            "weight": "./" + fp8_name,
                            "rows_per_expert": rows_per_expert,
                            "cols": 128,
                        },
                        "down": {
                            "weight": "./" + fp8_name,
                            "rows_per_expert": rows_per_expert,
                            "cols": 128,
                        },
                    }],
                },
            }

        fp8_package = str(Path(tmp) / "fp8-aligned.mollm")
        save_package(
            fp8_package, fp8_graph, fp8_graph, str(weights_dir),
            fp8_metadata(128))
        fp8_layer = read_package_metadata(fp8_package)[
            "moe_expert_storage"]["layers"][0]
        assert fp8_layer["gate_up"]["expert_data_bytes"] == 128 * 128
        assert fp8_layer["gate_up"]["expert_scales_bytes"] == 1
        assert fp8_layer["gate_up"]["groups_per_row"] == 1

        try:
            save_package(
                str(Path(tmp) / "fp8-misaligned.mollm"),
                fp8_graph, fp8_graph, str(weights_dir),
                fp8_metadata(64, 4))
        except ValueError as error:
            assert "128-row" in str(error)
        else:
            raise AssertionError(
                "misaligned FP8 expert scale tiles were accepted")

    print("MoE expert storage metadata tests passed")


if __name__ == "__main__":
    main()
