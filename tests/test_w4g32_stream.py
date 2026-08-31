#!/usr/bin/env python3
"""Verify offline streamed FP16-to-BG32 dense quantization."""

import io
import os
import tempfile
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from safetensors_stream import (
    quantize_w4g32_streamed_weight,
    quantize_w8pc_streamed_weight,
)
from transpile import (
    Precision,
    StreamedWeight,
    WEIGHT_FLAG_INT4_BG32,
    WeightByteRange,
)


def main():
    rows, columns = 10, 64
    values = np.empty((rows, columns), dtype=np.float16)
    for row in range(rows):
        for column in range(columns):
            values[row, column] = (
                ((row * 29 + column * 11) % 101 - 50) *
                (0.001 + 0.0003 * (column // 32)))

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "dense.f16"
        source.write_bytes(values.tobytes())
        original = StreamedWeight(
            precision=Precision.FP16,
            logical_shape=(rows, columns),
            data_ranges=[WeightByteRange(
                str(source), 0, values.nbytes)])
        quantized = quantize_w4g32_streamed_weight(original)
        assert quantized.precision == Precision.INT4
        assert quantized.flags == WEIGHT_FLAG_INT4_BG32
        assert quantized.group_size == 32
        assert quantized.num_groups == rows * 2

        output = io.BytesIO()
        quantized.write_to(output)
        payload = output.getvalue()[88:]
        assert len(payload) == 2 * 2 * 160

        for row in range(rows):
            for group in range(2):
                block = (row // 8) * 2 + group
                offset = block * 160
                scale = np.frombuffer(
                    payload, dtype="<f4", count=1,
                    offset=offset + (row % 8) * 4)[0]
                source_group = values[
                    row, group * 32:(group + 1) * 32].astype(np.float32)
                expected_scale = max(float(np.max(np.abs(source_group))) / 7, 1e-30)
                assert np.isclose(scale, expected_scale)
                packed = payload[
                    offset + 32 + (row % 8) * 16:
                    offset + 32 + (row % 8 + 1) * 16]
                decoded = np.empty(32, dtype=np.int8)
                for pair, byte in enumerate(packed):
                    low = byte & 0x0f
                    high = byte >> 4
                    decoded[2 * pair] = low if low < 8 else low - 16
                    decoded[2 * pair + 1] = high if high < 8 else high - 16
                expected = np.clip(
                    np.rint(source_group / expected_scale), -7, 7).astype(np.int8)
                np.testing.assert_array_equal(decoded, expected)

        w8 = quantize_w8pc_streamed_weight(original)
        assert w8.precision == Precision.INT8
        assert w8.group_size == columns
        assert w8.num_groups == rows
        output = io.BytesIO()
        w8.write_to(output)
        payload = output.getvalue()[88:]
        quantized_values = np.frombuffer(
            payload, dtype=np.int8, count=rows * columns
        ).reshape(rows, columns)
        scales = np.frombuffer(
            payload, dtype="<f4", count=rows, offset=rows * columns)
        source = values.astype(np.float32)
        expected_scales = np.max(np.abs(source), axis=1) / 127.0
        np.testing.assert_allclose(scales, expected_scales)
        np.testing.assert_array_equal(
            quantized_values,
            np.clip(
                np.rint(source / expected_scales[:, None]), -127, 127
            ).astype(np.int8))

    print("streamed W4G32 dense quantization tests passed")


if __name__ == "__main__":
    main()
