#!/usr/bin/env python3
"""Verify Qwen3.5-MoE streaming export preserves fused projection row order."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import qwen35_moe


def main():
    linear = "model.language_model.layers.0.linear_attn"
    attention = "model.language_model.layers.1.self_attn"
    parts = {
        f"{linear}.in_proj_qkv.weight":
            np.full((3, 2), 1, dtype=np.float16),
        f"{linear}.in_proj_a.weight":
            np.full((1, 2), 2, dtype=np.float16),
        f"{linear}.in_proj_b.weight":
            np.full((1, 2), 3, dtype=np.float16),
        f"{linear}.in_proj_z.weight":
            np.full((2, 2), 4, dtype=np.float16),
        f"{attention}.q_proj.weight":
            np.array([[5, 5], [8, 8], [6, 6], [9, 9]],
                     dtype=np.float16),
        f"{attention}.k_proj.weight":
            np.full((1, 2), 6, dtype=np.float16),
        f"{attention}.v_proj.weight":
            np.full((1, 2), 7, dtype=np.float16),
    }
    captured = {}

    old_selected = qwen35_moe._selected_safetensors
    old_required = qwen35_moe._required_weight_names
    old_write = qwen35_moe._write_weight_file
    try:
        qwen35_moe._required_weight_names = (
            lambda _tc, _layers: set(parts)
        )
        qwen35_moe._selected_safetensors = (
            lambda _model_dir, _wanted:
                ((name, "F16", value) for name, value in parts.items())
        )
        qwen35_moe._write_weight_file = (
            lambda path, data, **_kwargs:
                captured.__setitem__(os.path.basename(path), data.copy())
        )

        cfg = {
            "text_config": {
                "layer_types": ["linear_attention", "full_attention"],
                "num_attention_heads": 2,
                "head_dim": 1,
            }
        }
        with tempfile.TemporaryDirectory() as output_dir:
            qwen35_moe.export_weights(
                output_dir, output_dir, cfg, num_layers=2, quant="fp16")
    finally:
        qwen35_moe._selected_safetensors = old_selected
        qwen35_moe._required_weight_names = old_required
        qwen35_moe._write_weight_file = old_write

    linear_merged = captured[
        "model_language_model_layers_0_linear_attn_in_proj_weight.weights"
    ]
    attention_merged = captured[
        "model_language_model_layers_1_self_attn_qkv_proj_weight.weights"
    ]
    if not np.array_equal(
            linear_merged[:, 0],
            np.array([1, 1, 1, 2, 3, 4, 4], dtype=np.float16)):
        raise AssertionError("linear-attention fusion row order changed")
    if not np.array_equal(
            attention_merged[:, 0],
            np.array([5, 6, 8, 9, 6, 7], dtype=np.float16)):
        raise AssertionError("full-attention fusion row order changed")

    print("Qwen3.5-MoE projection fusion tests passed")


if __name__ == "__main__":
    main()
