"""Ascend MoE experts scaffolds; imports neither TileLang nor device code."""

from tileops._kernels.moe.ascend.backend import register_backend
from tileops._kernels.moe.ascend.fused_experts import AscendFusedMoEExpertsKernel
from tileops._kernels.moe.ascend.grouped_gemm import AscendMoEGroupedGEMMKernel
from tileops._kernels.moe.ascend.post_permute import AscendMoEPostPermuteKernel
from tileops._kernels.moe.ascend.pre_permute import AscendMoEPrePermuteKernel

__all__ = [
    "AscendFusedMoEExpertsKernel",
    "AscendMoEGroupedGEMMKernel",
    "AscendMoEPostPermuteKernel",
    "AscendMoEPrePermuteKernel",
    "register_backend",
]
