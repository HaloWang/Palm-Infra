#!/usr/bin/env python3
"""Structural tests for the staged Qwen4Exp text graph."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

from qwen4_exp import build_graph
from transpile import DimKind, OpType, propagate_dim_exprs


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def config():
    return {"text_config": {
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "rms_norm_eps": 1e-6,
        "hc_count": 4,
        "hc_lowrank": 8,
        "indexer_budget": 64,
        "rope_parameters": {"rope_theta": 1_000_000,
                            "partial_rotary_factor": 0.5},
        "head_dim": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "output_gate_type": "sigmoid",
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "vocab_size": 128,
        "ple_layer_ids": [],
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 101,
        "make_ngram_vocab_size_divisible_by": 8,
        "ple_embed_dim": 32,
        "ple_conv_kernel_size": 4,
        "eos_token_id": 127,
    }}


g = build_graph("/tmp/qwen4-test", config(), seq_len=1, n_ctx=64)
ops = [node.op_type for node in g._nodes]
check(ops.count(OpType.GROUP_RMS_NORM) == 5,
      "two GR reads per layer plus final mixer")
check(ops.count(OpType.GR_REDUCE) == 5,
      "all GR reads reduce four streams")
check(ops.count(OpType.GR_INJECT) == 4,
      "attention and MoE write back in every layer")
check(ops.count(OpType.GATED_DELTANET_CONV_DECODE) == 1,
      "linear layer reuses fused decode GDN")
gdn_nodes = [node for node in g._nodes
             if node.op_type == OpType.GATED_DELTANET_CONV_DECODE]
check((gdn_nodes[0].params_i32[4] & 2) != 0,
      "Qwen4Exp selects sigmoid rather than legacy SiLU output gating")
check(ops.count(OpType.SDPA) == 1,
      "short-context QSA uses equivalent gated GQA")
check(ops.count(OpType.MOE) == 2,
      "every decoder layer contains routed MoE")
check(ops.count(OpType.TILE) == 1,
      "embedding is widened into four residual streams once")

try:
    build_graph("/tmp/qwen4-test", config(), seq_len=1, n_ctx=65)
except ValueError as exc:
    check("QSA budget" in str(exc), "long-context error explains QSA limit")
else:
    raise AssertionError("long context must not silently run dense attention")

ple_cfg = config()
ple_cfg["text_config"]["ple_layer_ids"] = [2]
ple_graph = build_graph("/tmp/qwen4-test", ple_cfg, seq_len=1, n_ctx=64)
ple_ops = [node.op_type for node in ple_graph._nodes]
check(ple_ops.count(OpType.PLE_LOOKUP) == 1,
      "configured PLE layer performs one deterministic table lookup")
check(ple_ops.count(OpType.PLE_GATE) == 1,
      "PLE value is gated independently into residual streams")
check(ple_ops.count(OpType.PLE_DILATED_CONV) == 1,
      "PLE injects local lexical context with a dilated convolution")
aux_names = [node.params_str[0] for node in ple_graph._nodes
             if node.op_type == OpType.INPUT and node.params_str and
             node.params_str[0].startswith("aux_state")]
check(aux_names == ["aux_state0", "aux_state1"],
      "PLE owns token history and dilated-convolution state")

prefill_graph = build_graph(
    "/tmp/qwen4-test", ple_cfg, seq_len=8, n_ctx=64, is_prefill=True)
propagate_dim_exprs(prefill_graph._nodes)
for node in prefill_graph._nodes:
    if node.op_type in (
            OpType.GROUP_RMS_NORM, OpType.GR_REDUCE, OpType.GR_INJECT,
            OpType.PLE_LOOKUP, OpType.PLE_GATE, OpType.PLE_DILATED_CONV):
        check(node.dim_expr[1].kind == DimKind.SEQ,
              f"{node.op_type.name} must preserve runtime sequence length")

print("Qwen4Exp graph tests passed")
