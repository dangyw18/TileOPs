"""Lazy MoE exports keep Ascend backend loading independent of CUDA kernels."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tileops._kernels.moe.call_spec import (
        FusedTopKCall,
        FusedTopKFwdInterface,
        IndexedExpertCall,
        IndexedExpertDownFwdInterface,
        IndexedExpertGateUpFwdInterface,
        IndexedRouteStatsFwdInterface,
        IndexedWeightedReduceFwdInterface,
        MGroupedGEMMCall,
        MGroupedGEMMFP8Call,
        MGroupedGEMMFP8FwdInterface,
        MGroupedGEMMFwdInterface,
        PermuteAlignCall,
        PermuteAlignFwdInterface,
        PostPermuteCall,
        PostPermuteFwdInterface,
        PrePermuteCall,
        PrePermuteFwdInterface,
        SharedExpertMLPCall,
        SharedExpertMLPFwdInterface,
    )
    from tileops._kernels.moe.fused_topk import FusedTopKKernel
    from tileops._kernels.moe.indexed_expert_gemm import (
        IndexedExpertDownKernel,
        IndexedExpertGateUpKernel,
        IndexedExpertGEMMTemplate,
        IndexedRouteStatsKernel,
        IndexedWeightedReduceKernel,
    )
    from tileops._kernels.moe.moe_grouped_gemm import MoEGroupedGEMMKernel, MoEGroupedGEMMMMAKernel
    from tileops._kernels.moe.moe_grouped_gemm_fp8 import MoEGroupedGEMMFP8Kernel
    from tileops._kernels.moe.moe_grouped_gemm_fp8_mma import MoEGroupedGEMMFP8MMAKernel
    from tileops._kernels.moe.permute_align import MoEPermuteAlignKernel
    from tileops._kernels.moe.permute_contiguous import MoEPrePermuteContiguousKernel
    from tileops._kernels.moe.shared_expert_mlp import SharedExpertMLPKernel
    from tileops._kernels.moe.unpermute import MoEUnpermuteKernel

_LAZY = {
    "FusedTopKCall": ("tileops._kernels.moe.call_spec", "FusedTopKCall"),
    "FusedTopKFwdInterface": ("tileops._kernels.moe.call_spec", "FusedTopKFwdInterface"),
    "FusedTopKKernel": ("tileops._kernels.moe.fused_topk", "FusedTopKKernel"),
    "IndexedExpertCall": ("tileops._kernels.moe.call_spec", "IndexedExpertCall"),
    "IndexedExpertDownFwdInterface": (
        "tileops._kernels.moe.call_spec",
        "IndexedExpertDownFwdInterface",
    ),
    "IndexedExpertDownKernel": (
        "tileops._kernels.moe.indexed_expert_gemm",
        "IndexedExpertDownKernel",
    ),
    "IndexedExpertGEMMTemplate": (
        "tileops._kernels.moe.indexed_expert_gemm",
        "IndexedExpertGEMMTemplate",
    ),
    "IndexedExpertGateUpFwdInterface": (
        "tileops._kernels.moe.call_spec",
        "IndexedExpertGateUpFwdInterface",
    ),
    "IndexedExpertGateUpKernel": (
        "tileops._kernels.moe.indexed_expert_gemm",
        "IndexedExpertGateUpKernel",
    ),
    "IndexedRouteStatsFwdInterface": (
        "tileops._kernels.moe.call_spec",
        "IndexedRouteStatsFwdInterface",
    ),
    "IndexedRouteStatsKernel": (
        "tileops._kernels.moe.indexed_expert_gemm",
        "IndexedRouteStatsKernel",
    ),
    "IndexedWeightedReduceFwdInterface": (
        "tileops._kernels.moe.call_spec",
        "IndexedWeightedReduceFwdInterface",
    ),
    "IndexedWeightedReduceKernel": (
        "tileops._kernels.moe.indexed_expert_gemm",
        "IndexedWeightedReduceKernel",
    ),
    "MGroupedGEMMCall": ("tileops._kernels.moe.call_spec", "MGroupedGEMMCall"),
    "MGroupedGEMMFP8Call": ("tileops._kernels.moe.call_spec", "MGroupedGEMMFP8Call"),
    "MGroupedGEMMFP8FwdInterface": (
        "tileops._kernels.moe.call_spec",
        "MGroupedGEMMFP8FwdInterface",
    ),
    "MGroupedGEMMFwdInterface": ("tileops._kernels.moe.call_spec", "MGroupedGEMMFwdInterface"),
    "MoEGroupedGEMMFP8Kernel": (
        "tileops._kernels.moe.moe_grouped_gemm_fp8",
        "MoEGroupedGEMMFP8Kernel",
    ),
    "MoEGroupedGEMMFP8MMAKernel": (
        "tileops._kernels.moe.moe_grouped_gemm_fp8_mma",
        "MoEGroupedGEMMFP8MMAKernel",
    ),
    "MoEGroupedGEMMKernel": ("tileops._kernels.moe.moe_grouped_gemm", "MoEGroupedGEMMKernel"),
    "MoEGroupedGEMMMMAKernel": ("tileops._kernels.moe.moe_grouped_gemm", "MoEGroupedGEMMMMAKernel"),
    "MoEPermuteAlignKernel": ("tileops._kernels.moe.permute_align", "MoEPermuteAlignKernel"),
    "MoEPrePermuteContiguousKernel": (
        "tileops._kernels.moe.permute_contiguous",
        "MoEPrePermuteContiguousKernel",
    ),
    "MoEUnpermuteKernel": ("tileops._kernels.moe.unpermute", "MoEUnpermuteKernel"),
    "PermuteAlignCall": ("tileops._kernels.moe.call_spec", "PermuteAlignCall"),
    "PermuteAlignFwdInterface": ("tileops._kernels.moe.call_spec", "PermuteAlignFwdInterface"),
    "PostPermuteCall": ("tileops._kernels.moe.call_spec", "PostPermuteCall"),
    "PostPermuteFwdInterface": ("tileops._kernels.moe.call_spec", "PostPermuteFwdInterface"),
    "PrePermuteCall": ("tileops._kernels.moe.call_spec", "PrePermuteCall"),
    "PrePermuteFwdInterface": ("tileops._kernels.moe.call_spec", "PrePermuteFwdInterface"),
    "SharedExpertMLPCall": ("tileops._kernels.moe.call_spec", "SharedExpertMLPCall"),
    "SharedExpertMLPFwdInterface": (
        "tileops._kernels.moe.call_spec",
        "SharedExpertMLPFwdInterface",
    ),
    "SharedExpertMLPKernel": ("tileops._kernels.moe.shared_expert_mlp", "SharedExpertMLPKernel"),
}

__all__ = [
    "FusedTopKCall",
    "FusedTopKFwdInterface",
    "FusedTopKKernel",
    "IndexedExpertCall",
    "IndexedExpertDownFwdInterface",
    "IndexedExpertDownKernel",
    "IndexedExpertGEMMTemplate",
    "IndexedExpertGateUpFwdInterface",
    "IndexedExpertGateUpKernel",
    "IndexedRouteStatsFwdInterface",
    "IndexedRouteStatsKernel",
    "IndexedWeightedReduceFwdInterface",
    "IndexedWeightedReduceKernel",
    "MGroupedGEMMCall",
    "MGroupedGEMMFP8Call",
    "MGroupedGEMMFP8FwdInterface",
    "MGroupedGEMMFwdInterface",
    "MoEGroupedGEMMFP8Kernel",
    "MoEGroupedGEMMFP8MMAKernel",
    "MoEGroupedGEMMKernel",
    "MoEGroupedGEMMMMAKernel",
    "MoEPermuteAlignKernel",
    "MoEPrePermuteContiguousKernel",
    "MoEUnpermuteKernel",
    "PermuteAlignCall",
    "PermuteAlignFwdInterface",
    "PostPermuteCall",
    "PostPermuteFwdInterface",
    "PrePermuteCall",
    "PrePermuteFwdInterface",
    "SharedExpertMLPCall",
    "SharedExpertMLPFwdInterface",
    "SharedExpertMLPKernel",
]


def __getattr__(name: str) -> Any:
    import importlib

    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = entry
    value = getattr(importlib.import_module(module), symbol)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
