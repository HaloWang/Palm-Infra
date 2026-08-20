#!/usr/bin/env python3
"""Smoke tests for the HY-V3 checkpoint adapter and graph semantics."""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from converter import detect_model_type
from qwen3_moe import (
    ROUTER_SCORE_SIGMOID,
    _canonical_non_expert_name,
    _non_expert_weight_names,
    _prepare_config_for_weights,
    build_graph,
)
from transpile import OpType


def base_config():
    return {
        "model_type": "hy_v3",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 32,
        "moe_intermediate_size": 6,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "first_k_dense_replace": 1,
        "moe_router_use_sigmoid": True,
        "route_norm": True,
        "router_scaling_factor": 2.826,
        "vocab_size": 128,
        "rope_theta": 11158840.0,
        "rms_norm_eps": 1e-5,
    }


def checkpoint_names():
    names = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    for layer in range(2):
        pfx = f"model.layers.{layer}"
        names.update({
            f"{pfx}.input_layernorm.weight",
            f"{pfx}.post_attention_layernorm.weight",
            f"{pfx}.self_attn.q_proj.weight",
            f"{pfx}.self_attn.k_proj.weight",
            f"{pfx}.self_attn.v_proj.weight",
            f"{pfx}.self_attn.o_proj.weight",
            f"{pfx}.self_attn.q_norm.weight",
            f"{pfx}.self_attn.k_norm.weight",
        })
    names.update({
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.router.gate.weight",
        "model.layers.1.mlp.expert_bias",
        "model.layers.1.mlp.shared_mlp.gate_proj.weight",
        "model.layers.1.mlp.shared_mlp.up_proj.weight",
        "model.layers.1.mlp.shared_mlp.down_proj.weight",
        "model.layers.2.final_layernorm.weight",
    })
    return names


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = base_config()
        (root / "config.json").write_text(json.dumps(cfg))
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors"
                           for name in checkpoint_names()},
        }))

        assert detect_model_type(str(root)) == "hy_v3"
        cfg = _prepare_config_for_weights(root, cfg)

    assert cfg["_attn_norm_kind"] == "q_norm"
    assert cfg["_router_weight_suffix"] == "router.gate.weight"
    assert cfg["_router_bias_suffix"] == "expert_bias"
    assert cfg["_shared_expert_stem"] == "shared_mlp"
    assert cfg["_mtp_head_norm_suffix"] == "final_layernorm.weight"
    assert cfg["scoring_func"] == "sigmoid"
    assert cfg["norm_topk_prob"] is True
    assert cfg["routed_scaling_factor"] == 2.826
    assert cfg["n_shared_experts"] == 1

    selected = _non_expert_weight_names(cfg, 2, checkpoint_names())
    assert "model.layers.1.mlp.router.gate.weight" in selected
    assert "model.layers.1.mlp.expert_bias" in selected
    assert "model.layers.1.mlp.shared_mlp.gate_proj.weight" in selected
    assert "model.layers.1.mlp.gate.weight" not in selected

    assert _canonical_non_expert_name(
        "model.layers.1.mlp.router.gate.weight", cfg
    ) == "model.layers.1.mlp.gate.weight"
    assert _canonical_non_expert_name(
        "model.layers.1.mlp.expert_bias", cfg
    ) == "model.layers.1.mlp.gate.e_score_correction_bias"
    assert _canonical_non_expert_name(
        "model.layers.1.mlp.shared_mlp.up_proj.weight", cfg
    ) == "model.layers.1.mlp.shared_experts.up_proj.weight"
    assert _canonical_non_expert_name(
        "model.layers.2.final_layernorm.weight", cfg
    ) == "model.layers.2.shared_head.norm.weight"

    graph = build_graph(".", cfg, seq_len=4, n_ctx=32, is_prefill=True)
    assert graph.metadata["model_type"] == "hy_v3"
    moe_nodes = [node for node in graph._nodes if node.op_type == OpType.MOE]
    assert len(moe_nodes) == 1
    moe = moe_nodes[0]
    assert moe.params_i32[1] == 8
    assert moe.params_i32[2] == 2
    assert moe.params_i32[5] == ROUTER_SCORE_SIGMOID
    assert moe.params_i32[6] == 1  # normalize selected sigmoid weights
    assert moe.params_i32[7] == 0  # routed MoE takes the Metal fast path
    assert moe.params_i32[10] == 0  # shared expert has no separate gate
    assert moe.params_i32[11] == 4  # bias follows the routed expert weights
    assert len(moe.inputs) == 5
    assert abs(moe.params_f32[0] - 2.826) < 1e-6
    assert sum(node.op_type == OpType.SWIGLU for node in graph._nodes) == 2
    assert sum(node.op_type == OpType.ADD for node in graph._nodes) == 1

    print("HY-V3 adapter tests passed")


if __name__ == "__main__":
    main()
