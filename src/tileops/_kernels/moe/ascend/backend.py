"""Ascend MoE registration through the public whole-Op builder protocol.

FusedTopKFwdOp is intentionally absent. A colleague adds its builder to this
same target; FusedMoEFwdOp then uses the existing route_select composition.
"""

import torch

from tileops._kernels.moe.ascend.fused_experts import AscendFusedMoEExpertsKernel
from tileops._kernels.moe.ascend.grouped_gemm import AscendMoEGroupedGEMMKernel
from tileops._kernels.moe.ascend.post_permute import AscendMoEPostPermuteKernel
from tileops._kernels.moe.ascend.pre_permute import AscendMoEPrePermuteKernel
from tileops.backend import TensorSpec, register_detector, register_kernel_builder

__all__ = ["register_backend"]

_registered = False


def _check(spec: TensorSpec, name, shape=None, dtype=None, device=None):
    if spec.device.type not in ("npu", "privateuseone"):
        raise ValueError(f"{name} must be on an NPU")
    if shape is not None and tuple(spec.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}")
    if dtype is not None and spec.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if device is not None and spec.device != device:
        raise ValueError(f"{name} must be on {device}")


def _tight(layout):
    if getattr(layout, "selection_key", None) != "tight_physical_psum":
        raise NotImplementedError("Ascend MoE scaffold only declares tight_physical_psum layout")


def build_pre_permute(hidden_states, local_expert_ids, *, layout, num_local_experts):
    _tight(layout)
    _check(hidden_states, "hidden_states")
    tokens, hidden = hidden_states.shape
    _check(
        local_expert_ids,
        "local_expert_ids",
        (tokens, local_expert_ids.shape[1]),
        torch.int32,
        hidden_states.device,
    )
    return AscendMoEPrePermuteKernel(
        tokens, hidden, local_expert_ids.shape[1], num_local_experts, hidden_states.dtype
    )


def build_grouped_gemm(a, b, layout_metadata, bias, *, layout, activation=None, out_dtype=None):
    _tight(layout)
    _check(a, "a")
    experts, n, k = b.shape
    _check(a, "a", (a.shape[0], k))
    _check(b, "b", (experts, n, k), a.dtype, a.device)
    _check(layout_metadata, "layout_metadata", (experts,), torch.int32, a.device)
    if bias is not None:
        _check(bias, "bias", (experts, n), a.dtype, a.device)
    return AscendMoEGroupedGEMMKernel(
        a.shape[0],
        experts,
        n,
        k,
        a.dtype,
        activation=activation,
        out_dtype=out_dtype,
        has_bias=bias is not None,
    )


def build_post_permute(
    expert_output, topk_weights, inverse_indices, *, layout, epilogue=None, out_dtype=None
):
    _tight(layout)
    _check(expert_output, "expert_output")
    tokens, top_k = topk_weights.shape
    _check(topk_weights, "topk_weights", (tokens, top_k), torch.float32, expert_output.device)
    _check(inverse_indices, "inverse_indices", (tokens * top_k,), torch.int32, expert_output.device)
    scale = 1.0 if epilogue is None else epilogue.routed_scaling_factor
    return AscendMoEPostPermuteKernel(
        tokens,
        top_k,
        expert_output.shape[1],
        expert_output.dtype,
        routed_scaling_factor=scale,
        out_dtype=out_dtype,
        materialized_rows=expert_output.shape[0],
    )


def build_experts(
    hidden_states,
    w_gate_up,
    w_down,
    topk_weights,
    topk_ids,
    b_gate_up,
    b_down,
    *,
    routed_scaling_factor=1.0,
    activation="silu_and_mul",
):
    _check(hidden_states, "hidden_states")
    tokens, hidden = hidden_states.shape
    experts, twice_ffn, weight_hidden = w_gate_up.shape
    if twice_ffn % 2 or weight_hidden != hidden:
        raise ValueError("w_gate_up must have shape [E,2F,H]")
    top_k = topk_ids.shape[1]
    _check(
        w_gate_up,
        "w_gate_up",
        (experts, twice_ffn, hidden),
        hidden_states.dtype,
        hidden_states.device,
    )
    _check(
        w_down,
        "w_down",
        (experts, hidden, twice_ffn // 2),
        hidden_states.dtype,
        hidden_states.device,
    )
    _check(topk_weights, "topk_weights", (tokens, top_k), torch.float32, hidden_states.device)
    _check(topk_ids, "topk_ids", (tokens, top_k), torch.int32, hidden_states.device)
    if b_gate_up is not None:
        _check(
            b_gate_up, "b_gate_up", (experts, twice_ffn), hidden_states.dtype, hidden_states.device
        )
    if b_down is not None:
        _check(b_down, "b_down", (experts, hidden), hidden_states.dtype, hidden_states.device)
    return AscendFusedMoEExpertsKernel(
        tokens,
        hidden,
        twice_ffn // 2,
        experts,
        top_k,
        hidden_states.dtype,
        routed_scaling_factor=routed_scaling_factor,
        activation=activation,
        has_gate_up_bias=b_gate_up is not None,
        has_down_bias=b_down is not None,
    )


def register_backend():
    """Register once; own only the experts primitives, never the TopK builder."""
    global _registered
    if _registered:
        return
    register_detector(
        target="ascend", detect=lambda device: device.type in ("npu", "privateuseone")
    )
    for op, builder in (
        ("MoEPrePermuteFwdOp", build_pre_permute),
        ("MoEGroupedGEMMFwdOp", build_grouped_gemm),
        ("MoEPostPermuteFwdOp", build_post_permute),
        ("FusedMoEExpertsFwdOp", build_experts),
    ):
        register_kernel_builder(op=op, target="ascend", build_kernel=builder)
    _registered = True


register_backend()
