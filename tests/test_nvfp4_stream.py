#!/usr/bin/env python3
"""Verify streamed NVFP4 expert pair packing and XMAP flags."""

import io
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from safetensors_stream import SafeTensorSlice, aggregate_nvfp4_experts
from transpile import (
    WEIGHT_FLAG_EXPERT_INTERLEAVED,
    WEIGHT_FLAG_NVFP4_Q8_PAIR,
    WEIGHT_HEADER_STRUCT,
    WeightByteRange,
)


class FakeIndex:
    def __init__(self, tensors):
        self.tensors = tensors

    def tensor(self, name):
        return self.tensors[name]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        source_path = Path(tmp) / "nvfp4.bin"
        payload = bytearray()
        tensors = {}

        def add(name, dtype, shape, data):
            offset = len(payload)
            payload.extend(data)
            tensors[name] = SafeTensorSlice(
                name, dtype, shape,
                WeightByteRange(str(source_path), offset, len(data)))

        original = []
        block_scales = []
        globals_ = []
        for expert in range(2):
            values = (
                np.arange(4 * 16, dtype=np.uint8) + expert * 64
            ).reshape(4, 16)
            scales = np.arange(4 * 2, dtype=np.uint8) + 0x30 + expert * 8
            global_scale = struct.pack("<f", 1.0 + expert)
            name = f"expert.{expert}.weight"
            add(name, "U8", (4, 16), values.tobytes())
            add(name + "_scale", "F8_E4M3", (4, 2), scales.tobytes())
            add(name + "_scale_2", "F32", (), global_scale)
            original.append(values)
            block_scales.append(scales.tobytes())
            globals_.append(global_scale)

        source_path.write_bytes(payload)
        streamed = aggregate_nvfp4_experts(
            FakeIndex(tensors),
            [[f"expert.{expert}.weight"] for expert in range(2)])
        header = streamed.header()
        assert header["flags"] & WEIGHT_FLAG_EXPERT_INTERLEAVED
        assert header["flags"] & WEIGHT_FLAG_NVFP4_Q8_PAIR

        output = io.BytesIO()
        streamed.write_to(output)
        raw = output.getvalue()
        unpacked = WEIGHT_HEADER_STRUCT.unpack(raw[:88])
        assert unpacked[1] == header["flags"]

        expected = bytearray()
        for expert in range(2):
            values = original[expert]
            pair_packed = np.ascontiguousarray(
                values.reshape(4, 2, 8).transpose(1, 0, 2)
            ).tobytes()
            expected.extend(pair_packed)
            expected.extend(block_scales[expert])
            expected.extend(globals_[expert] * 4)
        assert raw[88:] == expected

        # Exercise aligned streaming with a deliberately non-aligned generic
        # chunk size; every transform call must still contain whole row tiles.
        aligned = io.BytesIO()
        streamed._copy_ranges(aligned, streamed.data_ranges, chunk_size=37)
        expected_data = bytearray()
        for values in original:
            expected_data.extend(np.ascontiguousarray(
                values.reshape(4, 2, 8).transpose(1, 0, 2)
            ).tobytes())
        assert aligned.getvalue() == expected_data

    print("NVFP4 streamed pair-pack tests passed")


if __name__ == "__main__":
    main()
