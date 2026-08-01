"""Selective, streaming access to sharded safetensors checkpoints.

This module deliberately does not depend on PyTorch or safetensors. It parses
the small JSON headers and exposes byte ranges that the package writer can
copy directly into a `.mollm` file.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from transpile import (
    Precision,
    StreamedWeight,
    WEIGHT_FLAG_FP8_BLOCK128,
    WeightByteRange,
)


_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "I32": 4,
    "I64": 8,
}


@dataclass(frozen=True)
class SafeTensorSlice:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source: WeightByteRange

    @property
    def nbytes(self) -> int:
        return self.source.size


class SafeTensorIndex:
    """Index tensor byte ranges without materializing checkpoint tensors."""

    def __init__(self, model_dir: str | Path):
        self.root = Path(model_dir)
        self._headers: dict[str, tuple[int, dict]] = {}
        index_path = self.root / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            self.weight_map = dict(index["weight_map"])
        else:
            files = sorted(self.root.glob("*.safetensors"))
            if len(files) != 1:
                raise FileNotFoundError(
                    "expected model.safetensors.index.json or one "
                    "safetensors file")
            self.weight_map = {}
            header = self._read_shard_header(files[0].name)
            for name in header:
                if name != "__metadata__":
                    self.weight_map[name] = files[0].name

    def _read_shard_header(self, shard: str) -> dict:
        path = self.root / shard
        with path.open("rb") as source:
            raw_size = source.read(8)
            if len(raw_size) != 8:
                raise ValueError(f"truncated safetensors header: {path}")
            header_size = struct.unpack("<Q", raw_size)[0]
            if header_size <= 0 or header_size > (1 << 30):
                raise ValueError(
                    f"invalid safetensors header size in {path}")
            raw_header = source.read(header_size)
            if len(raw_header) != header_size:
                raise ValueError(f"truncated safetensors header: {path}")
        header = json.loads(raw_header)
        self._headers[shard] = (8 + header_size, header)
        return header

    def tensor(self, name: str) -> SafeTensorSlice:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"checkpoint tensor not found: {name}")
        if shard not in self._headers:
            self._read_shard_header(shard)
        data_base, header = self._headers[shard]
        entry = header[name]
        dtype = entry["dtype"]
        shape = tuple(int(dim) for dim in entry["shape"])
        begin, end = (int(value) for value in entry["data_offsets"])
        if begin < 0 or end < begin:
            raise ValueError(f"invalid byte range for {name}")
        item_size = _DTYPE_BYTES.get(dtype)
        if item_size is None:
            raise ValueError(f"unsupported safetensors dtype {dtype}: {name}")
        expected = item_size
        for dim in shape:
            expected *= dim
        if expected != end - begin:
            raise ValueError(
                f"safetensors byte-size mismatch for {name}: "
                f"{end - begin} != {expected}")
        return SafeTensorSlice(
            name=name,
            dtype=dtype,
            shape=shape,
            source=WeightByteRange(
                str(self.root / shard), data_base + begin, end - begin),
        )

    def contains(self, name: str) -> bool:
        return name in self.weight_map


def _bf16_to_fp16(chunk: bytes) -> bytes:
    if len(chunk) % 2:
        raise ValueError("BF16 stream chunk is not element-aligned")
    bf16 = np.frombuffer(chunk, dtype="<u2")
    fp32_bits = bf16.astype(np.uint32) << 16
    return fp32_bits.view(np.float32).astype(np.float16).tobytes()


def _bf16_to_fp32(chunk: bytes) -> bytes:
    if len(chunk) % 2:
        raise ValueError("BF16 stream chunk is not element-aligned")
    bf16 = np.frombuffer(chunk, dtype="<u2")
    return (bf16.astype(np.uint32) << 16).view(np.float32).tobytes()


def _fp16_to_fp32(chunk: bytes) -> bytes:
    if len(chunk) % 2:
        raise ValueError("FP16 stream chunk is not element-aligned")
    return np.frombuffer(chunk, dtype="<f2").astype(np.float32).tobytes()


def _int64_to_int32(chunk: bytes) -> bytes:
    if len(chunk) % 8:
        raise ValueError("INT64 stream chunk is not element-aligned")
    values = np.frombuffer(chunk, dtype="<i8")
    if np.any(values < np.iinfo(np.int32).min) or np.any(
            values > np.iinfo(np.int32).max):
        raise OverflowError("INT64 routing value does not fit INT32")
    return values.astype("<i4").tobytes()


def dense_streamed_weight(index: SafeTensorIndex,
                          name: str) -> StreamedWeight:
    """Create a streamed dense weight, preserving native quantization."""
    tensor = index.tensor(name)
    if tensor.dtype in ("BF16", "F16", "F32"):
        precision = {
            "BF16": Precision.FP16,
            "F16": Precision.FP16,
            "F32": Precision.FP32,
        }[tensor.dtype]
        transform = _bf16_to_fp16 if tensor.dtype == "BF16" else None
        data_range = WeightByteRange(
            tensor.source.path, tensor.source.offset,
            tensor.source.size, transform)
        return StreamedWeight(
            precision=precision, logical_shape=tensor.shape,
            data_ranges=[data_range])

    scale = index.tensor(name.rsplit(".", 1)[0] + ".scale")
    if scale.dtype != "F8_E8M0":
        raise ValueError(f"native quantized scale is not E8M0: {name}")
    if tensor.dtype == "F8_E4M3":
        if len(tensor.shape) != 2:
            raise ValueError(f"FP8 weight must be a matrix: {name}")
        n, k = tensor.shape
        expected_scale = ((n + 127) // 128, (k + 127) // 128)
        if scale.shape != expected_scale:
            raise ValueError(
                f"FP8 scale shape mismatch for {name}: "
                f"{scale.shape} != {expected_scale}")
        return StreamedWeight(
            precision=Precision.FP8_E4M3,
            logical_shape=tensor.shape,
            data_ranges=[tensor.source],
            scale_ranges=[scale.source],
            group_size=128,
            num_groups=scale.nbytes,
            flags=WEIGHT_FLAG_FP8_BLOCK128,
        )
    if tensor.dtype == "I8":
        if len(tensor.shape) != 2:
            raise ValueError(f"MXFP4 weight must be a matrix: {name}")
        n, packed_k = tensor.shape
        logical_shape = (n, packed_k * 2)
        expected_scale = (n, logical_shape[1] // 32)
        if scale.shape != expected_scale:
            raise ValueError(
                f"MXFP4 scale shape mismatch for {name}: "
                f"{scale.shape} != {expected_scale}")
        return StreamedWeight(
            precision=Precision.MXFP4,
            logical_shape=logical_shape,
            data_ranges=[tensor.source],
            scale_ranges=[scale.source],
            group_size=32,
            num_groups=scale.nbytes,
        )
    raise ValueError(f"unsupported dense dtype {tensor.dtype}: {name}")


def fp32_streamed_weight(
        index: SafeTensorIndex, name: str,
        logical_shape: tuple[int, ...] | None = None) -> StreamedWeight:
    """Stream an unquantized tensor as FP32.

    DeepSeek-V4 stores several numerically sensitive RMSNorm/compressor
    parameters as BF16 even though its reference modules materialize them as
    FP32. Conversion happens one package-writer chunk at a time.
    """
    tensor = index.tensor(name)
    shape = logical_shape or tensor.shape
    if np.prod(shape, dtype=np.int64) != np.prod(
            tensor.shape, dtype=np.int64):
        raise ValueError(
            f"logical FP32 shape changes element count: "
            f"{tensor.shape} -> {shape}")
    if tensor.dtype == "F32":
        data_range = tensor.source
    elif tensor.dtype == "BF16":
        data_range = WeightByteRange(
            tensor.source.path, tensor.source.offset, tensor.source.size,
            _bf16_to_fp32, tensor.source.size * 2)
    elif tensor.dtype == "F16":
        data_range = WeightByteRange(
            tensor.source.path, tensor.source.offset, tensor.source.size,
            _fp16_to_fp32, tensor.source.size * 2)
    else:
        raise ValueError(f"cannot convert {tensor.dtype} to FP32: {name}")
    return StreamedWeight(
        precision=Precision.FP32,
        logical_shape=tuple(int(dim) for dim in shape),
        data_ranges=[data_range])


def integer_streamed_weight(index: SafeTensorIndex,
                            name: str,
                            logical_shape: tuple[int, ...] | None = None
                            ) -> StreamedWeight:
    tensor = index.tensor(name)
    if tensor.dtype == "I32":
        data_range = tensor.source
    elif tensor.dtype == "I64":
        data_range = WeightByteRange(
            tensor.source.path, tensor.source.offset, tensor.source.size,
            _int64_to_int32, tensor.source.size // 2)
    else:
        raise ValueError(f"integer tensor must be I32/I64: {name}")
    return StreamedWeight(
        precision=Precision.INT32,
        logical_shape=logical_shape or tensor.shape,
        data_ranges=[data_range])


def aggregate_mxfp4_experts(index: SafeTensorIndex,
                            names: Iterable[str], *,
                            interleave_expert_count: int = 0) -> StreamedWeight:
    """Concatenate per-expert MXFP4 matrices in expert-major row order."""
    tensors = [index.tensor(name) for name in names]
    if not tensors:
        raise ValueError("cannot aggregate an empty expert list")
    first = tensors[0]
    if first.dtype != "I8" or len(first.shape) != 2:
        raise ValueError(f"expert is not packed MXFP4: {first.name}")
    rows, packed_k = first.shape
    for tensor in tensors:
        if tensor.dtype != "I8" or tensor.shape != first.shape:
            raise ValueError(
                f"inconsistent MXFP4 expert shape: {tensor.name}")

    scales = [
        index.tensor(tensor.name.rsplit(".", 1)[0] + ".scale")
        for tensor in tensors
    ]
    expected_scale = (rows, packed_k * 2 // 32)
    for scale in scales:
        if scale.dtype != "F8_E8M0" or scale.shape != expected_scale:
            raise ValueError(
                f"inconsistent MXFP4 expert scale: {scale.name}")
    return StreamedWeight(
        precision=Precision.MXFP4,
        logical_shape=(len(tensors) * rows, packed_k * 2),
        data_ranges=[tensor.source for tensor in tensors],
        scale_ranges=[scale.source for scale in scales],
        group_size=32,
        num_groups=sum(scale.nbytes for scale in scales),
        expert_interleave_count=interleave_expert_count,
    )
