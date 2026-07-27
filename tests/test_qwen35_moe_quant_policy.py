#!/usr/bin/env python3
"""Unit tests for Qwen3.5-MoE converter quantization policy."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from qwen35_moe import _quant_spec


def check(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected}, got {actual}")


check(_quant_spec("w4g32", 2048, "lm_head.weight"), ("w4", 32),
      "pure W4G32 does not promote tensors")
check(_quant_spec("w4mixg128", 2048, "lm_head.weight"), ("w8", 128),
      "mixed W4G128 promotes lm_head")
check(_quant_spec("w4mixg32", 2048, "lm_head.weight"), ("w8", 32),
      "mixed W4G32 reuses the lm_head promotion")
check(_quant_spec(
    "w4mixg32", 2048,
    "model.language_model.layers.0.self_attn.qkv_proj.weight"),
    ("w8", 32), "mixed W4G32 promotes fused attention QKV")
check(_quant_spec(
    "w4mixg32", 2048,
    "model.language_model.layers.0.mlp.experts.gate_up_proj.weight"),
    ("w4", 32), "mixed W4G32 keeps expert gate/up at W4")
check(_quant_spec(
    "w4mixg32", 1024,
    "model.language_model.layers.0.mlp.experts.down_proj.weight"),
    ("w4", 32), "mixed W4G32 keeps expert down at W4")

print("Qwen3.5-MoE quant policy tests passed")
