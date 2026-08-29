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
    "U8": 1,
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


def _bf16_to_fp32_plus_one(chunk: bytes) -> bytes:
    values = np.frombuffer(_bf16_to_fp32(chunk), dtype="<f4").copy()
    values += np.float32(1.0)
    return values.tobytes()


def _fp16_to_fp32_plus_one(chunk: bytes) -> bytes:
    values = np.frombuffer(chunk, dtype="<f2").astype(np.float32)
    values += np.float32(1.0)
    return values.tobytes()


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


def fp32_plus_one_streamed_weight(
        index: SafeTensorIndex, name: str,
        logical_shape: tuple[int, ...] | None = None) -> StreamedWeight:
    """Stream a parameter whose module uses ``1 + weight`` as FP32."""
    tensor = index.tensor(name)
    shape = logical_shape or tensor.shape
    if np.prod(shape, dtype=np.int64) != np.prod(
            tensor.shape, dtype=np.int64):
        raise ValueError(
            f"logical FP32 shape changes element count: "
            f"{tensor.shape} -> {shape}")
    if tensor.dtype == "BF16":
        transform = _bf16_to_fp32_plus_one
        output_size = tensor.source.size * 2
    elif tensor.dtype == "F16":
        transform = _fp16_to_fp32_plus_one
        output_size = tensor.source.size * 2
    elif tensor.dtype == "F32":
        def transform(chunk: bytes) -> bytes:
            values = np.frombuffer(chunk, dtype="<f4").copy()
            values += np.float32(1.0)
            return values.tobytes()
        output_size = tensor.source.size
    else:
        raise ValueError(f"cannot convert {tensor.dtype} to FP32: {name}")
    return StreamedWeight(
        precision=Precision.FP32,
        logical_shape=tuple(int(dim) for dim in shape),
        data_ranges=[WeightByteRange(
            tensor.source.path, tensor.source.offset, tensor.source.size,
            transform, output_size)])


def concatenate_fp16_streamed_weights(
        index: SafeTensorIndex, names: Iterable[str],
        logical_shape: tuple[int, ...]) -> StreamedWeight:
    """Row-concatenate BF16/F16 tensors into one streamed FP16 tensor."""
    ranges = []
    elements = 0
    for name in names:
        tensor = index.tensor(name)
        if tensor.dtype not in ("BF16", "F16"):
            raise ValueError(f"cannot concatenate {tensor.dtype} as FP16: {name}")
        elements += int(np.prod(tensor.shape, dtype=np.int64))
        ranges.append(WeightByteRange(
            tensor.source.path, tensor.source.offset, tensor.source.size,
            _bf16_to_fp16 if tensor.dtype == "BF16" else None))
    expected = int(np.prod(logical_shape, dtype=np.int64))
    if elements != expected:
        raise ValueError(
            f"concatenated FP16 element count mismatch: {elements} != {expected}")
    return StreamedWeight(
        precision=Precision.FP16,
        logical_shape=tuple(int(dim) for dim in logical_shape),
        data_ranges=ranges)


def concatenate_fp16_row_slices(
        index: SafeTensorIndex,
        slices: Iterable[tuple[str, int, int]],
        logical_shape: tuple[int, ...]) -> StreamedWeight:
    """Concatenate contiguous row ranges from rank-2 tensors as FP16."""
    ranges = []
    elements = 0
    for name, row_begin, row_count in slices:
        tensor = index.tensor(name)
        if tensor.dtype not in ("BF16", "F16") or len(tensor.shape) != 2:
            raise ValueError(f"row slice is not a BF16/F16 matrix: {name}")
        rows, cols = tensor.shape
        if row_begin < 0 or row_count <= 0 or row_begin + row_count > rows:
            raise ValueError(
                f"invalid row slice [{row_begin}, {row_begin + row_count}) "
                f"for {name} with {rows} rows")
        row_bytes = cols * 2
        ranges.append(WeightByteRange(
            tensor.source.path,
            tensor.source.offset + row_begin * row_bytes,
            row_count * row_bytes,
            _bf16_to_fp16 if tensor.dtype == "BF16" else None))
        elements += row_count * cols
    expected = int(np.prod(logical_shape, dtype=np.int64))
    if elements != expected:
        raise ValueError(
            f"concatenated FP16 slice count mismatch: {elements} != {expected}")
    return StreamedWeight(
        precision=Precision.FP16,
        logical_shape=tuple(int(dim) for dim in logical_shape),
        data_ranges=ranges)


def raw_u8_streamed_weight(
        index: SafeTensorIndex, names: Iterable[str],
        logical_shape: tuple[int, ...], *,
        accepted_dtypes: tuple[str, ...] = ("U8", "F8_E4M3"),
        ) -> StreamedWeight:
    """Concatenate byte-exact tensors used by lookup tables."""
    tensors = [index.tensor(name) for name in names]
    for tensor in tensors:
        if tensor.dtype not in accepted_dtypes:
            raise ValueError(
                f"raw tensor has unsupported dtype {tensor.dtype}: "
                f"{tensor.name}")
    actual = sum(tensor.nbytes for tensor in tensors)
    expected = int(np.prod(logical_shape, dtype=np.int64))
    if actual != expected:
        raise ValueError(f"raw byte count mismatch: {actual} != {expected}")
    return StreamedWeight(
        precision=Precision.RAW_U8,
        logical_shape=tuple(int(dim) for dim in logical_shape),
        data_ranges=[tensor.source for tensor in tensors])


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


def _repeat_f32_scalar(count: int):
    def transform(chunk: bytes) -> bytes:
        if len(chunk) != 4:
            raise ValueError("NVFP4 global scale must be one FP32 scalar")
        return chunk * count
    return transform


def aggregate_nvfp4_experts(
        index: SafeTensorIndex,
        expert_projection_names: Iterable[Iterable[str]]) -> StreamedWeight:
    """Fuse and aggregate native NVFP4 expert projections.

    Each inner iterable contains projections that are row-concatenated for a
    single expert (gate + up, or just down). Packed E2M1 bytes stay native.
    The sidecar stores all E4M3 block scales followed by one FP32 global scale
    per output row, allowing fused projections to retain distinct globals.
    """
    experts = [list(names) for names in expert_projection_names]
    if not experts or not experts[0]:
        raise ValueError("cannot aggregate empty NVFP4 experts")

    data_ranges = []
    scale_ranges = []
    rows_per_expert = None
    logical_k = None
    total_groups = 0
    for expert_names in experts:
        expert_rows = 0
        expert_scales = []
        expert_globals = []
        for name in expert_names:
            tensor = index.tensor(name)
            if tensor.dtype != "U8" or len(tensor.shape) != 2:
                raise ValueError(f"expert is not packed NVFP4: {name}")
            rows, packed_k = tensor.shape
            current_k = packed_k * 2
            if current_k % 16 != 0:
                raise ValueError(f"NVFP4 K is not block-16 aligned: {name}")
            if logical_k is None:
                logical_k = current_k
            elif logical_k != current_k:
                raise ValueError(f"inconsistent NVFP4 K dimension: {name}")
            scale = index.tensor(name + "_scale")
            scale2 = index.tensor(name + "_scale_2")
            expected_scale = (rows, current_k // 16)
            if scale.dtype != "F8_E4M3" or scale.shape != expected_scale:
                raise ValueError(f"invalid NVFP4 block scale: {scale.name}")
            if scale2.dtype != "F32" or scale2.shape not in ((), (1,)):
                raise ValueError(f"invalid NVFP4 global scale: {scale2.name}")
            data_ranges.append(tensor.source)
            expert_scales.append(scale.source)
            expert_globals.append(WeightByteRange(
                scale2.source.path, scale2.source.offset, scale2.source.size,
                _repeat_f32_scalar(rows), rows * 4))
            expert_rows += rows
            total_groups += rows * (current_k // 16)
        if rows_per_expert is None:
            rows_per_expert = expert_rows
        elif rows_per_expert != expert_rows:
            raise ValueError("NVFP4 experts have inconsistent fused row counts")
        # The runtime sidecar is [all block scales][one global per row].
        scale_ranges.extend(expert_scales)
        scale_ranges.extend(expert_globals)

    assert rows_per_expert is not None and logical_k is not None
    return StreamedWeight(
        precision=Precision.NVFP4,
        logical_shape=(len(experts) * rows_per_expert, logical_k),
        data_ranges=data_ranges,
        scale_ranges=scale_ranges,
        group_size=16,
        num_groups=total_groups,
        expert_interleave_count=len(experts),
    )
