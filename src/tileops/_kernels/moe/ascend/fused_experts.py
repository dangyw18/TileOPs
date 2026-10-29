"""Experts composition: pre-permute -> gate/up+activation -> down -> post-permute.

This is a host composition, not one device kernel. All token counts use the same
grouped path; CUDA's small-route indexed kernels are deliberately not selected.
"""

import torch

from tileops._kernels.moe.ascend.common import (
    check_activation,
    check_dtype,
    check_out,
    check_scale,
    check_tensor,
    example_parser,
    gated_activation,
    random_tensor,
    run_example,
)
from tileops._kernels.moe.ascend.grouped_gemm import AscendMoEGroupedGEMMKernel
from tileops._kernels.moe.ascend.post_permute import AscendMoEPostPermuteKernel
from tileops._kernels.moe.ascend.pre_permute import AscendMoEPrePermuteKernel

__all__ = ["AscendFusedMoEExpertsKernel"]


class AscendFusedMoEExpertsKernel:
    def __init__(
        self,
        tokens,
        hidden,
        ffn,
        experts,
        top_k,
        dtype=torch.bfloat16,
        *,
        routed_scaling_factor=1.0,
        activation="silu_and_mul",
        has_gate_up_bias=False,
        has_down_bias=False,
    ):
        check_activation(activation)
        if activation is None:
            raise ValueError("experts require a gated activation")
        rows = tokens * top_k
        self.pre_permute = AscendMoEPrePermuteKernel(tokens, hidden, top_k, experts, dtype)
        self.gate_up = AscendMoEGroupedGEMMKernel(
            rows, experts, 2 * ffn, hidden, dtype, activation=activation, has_bias=has_gate_up_bias
        )
        self.down = AscendMoEGroupedGEMMKernel(
            rows, experts, hidden, ffn, dtype, has_bias=has_down_bias
        )
        self.post_permute = AscendMoEPostPermuteKernel(
            tokens, top_k, hidden, dtype, routed_scaling_factor=routed_scaling_factor
        )
        self.tokens, self.hidden, self.dtype = tokens, hidden, dtype
        self.config = {"path": "tight_grouped", "device_bodies": "not_implemented"}

    @property
    def implemented(self):
        return all(
            k.implemented for k in (self.pre_permute, self.gate_up, self.down, self.post_permute)
        )

    def __call__(
        self,
        hidden_states,
        w_gate_up,
        w_down,
        topk_weights,
        topk_ids,
        b_gate_up=None,
        b_down=None,
        *,
        out=None,
    ):
        check_out(
            out,
            (self.tokens, self.hidden),
            self.dtype,
            hidden_states.device,
            (hidden_states, w_gate_up, w_down, topk_weights, topk_ids, b_gate_up, b_down),
        )
        packed, physical_ends, inverse = self.pre_permute(hidden_states, topk_ids)
        activated = self.gate_up(packed, w_gate_up, physical_ends, b_gate_up)
        expert_output = self.down(activated, w_down, physical_ends, b_down)
        return self.post_permute(expert_output, topk_weights, inverse, out=out)

    def request_tune(self):
        raise NotImplementedError("Ascend MoE tuning is not implemented")


def torch_reference(
    hidden_states,
    w_gate_up,
    w_down,
    topk_weights,
    topk_ids,
    b_gate_up=None,
    b_down=None,
    *,
    routed_scaling_factor=1.0,
    activation="silu_and_mul",
):
    """FP32 expert math and reduction, rounded once, like ref_routed_experts.

    No [T,K,H,F] weight expansion is formed. These Torch operations are an oracle
    only and are never called by AscendFusedMoEExpertsKernel.__call__.
    """
    tokens, hidden = hidden_states.shape
    experts, twice_ffn, weight_hidden = w_gate_up.shape
    top_k = topk_ids.shape[1]
    check_dtype(hidden_states.dtype)
    check_scale(routed_scaling_factor)
    check_activation(activation)
    if activation is None or twice_ffn % 2 or weight_hidden != hidden or experts <= 0:
        raise ValueError("invalid experts dimensions or activation")
    check_tensor(
        w_gate_up,
        "w_gate_up",
        (experts, twice_ffn, hidden),
        hidden_states.dtype,
        hidden_states.device,
    )
    check_tensor(
        w_down,
        "w_down",
        (experts, hidden, twice_ffn // 2),
        hidden_states.dtype,
        hidden_states.device,
    )
    check_tensor(topk_weights, "topk_weights", (tokens, top_k), torch.float32, hidden_states.device)
    check_tensor(topk_ids, "topk_ids", (tokens, top_k), torch.int32, hidden_states.device)
    if b_gate_up is not None:
        check_tensor(
            b_gate_up, "b_gate_up", (experts, twice_ffn), hidden_states.dtype, hidden_states.device
        )
    if b_down is not None:
        check_tensor(b_down, "b_down", (experts, hidden), hidden_states.dtype, hidden_states.device)
    if bool(((topk_ids < 0) | (topk_ids >= experts)).any()):
        raise ValueError("expert IDs must be local and in range")
    output = torch.zeros((tokens, hidden), dtype=torch.float32, device=hidden_states.device)
    for expert in range(experts):
        token, route = (topk_ids == expert).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        gate_up = hidden_states[token].float() @ w_gate_up[expert].float().T
        if b_gate_up is not None:
            gate_up = gate_up + b_gate_up[expert].float()
        down = gated_activation(gate_up, activation) @ w_down[expert].float().T
        if b_down is not None:
            down = down + b_down[expert].float()
        output.index_add_(0, token, down * topk_weights[token, route].unsqueeze(-1))
    return (output * routed_scaling_factor).to(hidden_states.dtype)


def main():
    parser = example_parser(__doc__)
    parser.add_argument(
        "--activation",
        choices=("silu_and_mul", "gelu_and_mul", "swigluoai"),
        default="silu_and_mul",
    )
    parser.add_argument("--bias", action="store_true")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cpu").manual_seed(42)
    hidden, w13, w2 = (
        random_tensor(shape, dtype, generator) for shape in ((4, 32), (4, 32, 32), (4, 32, 16))
    )
    weights = torch.randn((4, 2), generator=generator, dtype=torch.float32).softmax(-1)
    ids = torch.tensor([[0, 2], [2, 0], [0, 0], [2, 2]], dtype=torch.int32)
    b13 = random_tensor((4, 32), dtype, generator) if args.bias else None
    b2 = random_tensor((4, 32), dtype, generator) if args.bias else None
    inputs = (hidden, w13, w2, weights, ids, b13, b2)
    expected = torch_reference(*inputs, activation=args.activation, routed_scaling_factor=2.5)
    kernel = AscendFusedMoEExpertsKernel(
        4,
        32,
        16,
        4,
        2,
        dtype,
        activation=args.activation,
        routed_scaling_factor=2.5,
        has_gate_up_bias=args.bias,
        has_down_bias=args.bias,
    )

    def invoke(*tensors):
        from tileops.moe import FusedMoEExpertsFwdOp

        op = FusedMoEExpertsFwdOp(2.5, activation=args.activation, target="ascend")
        return op(*tensors)

    run_example(invoke, inputs, expected, args, implemented=kernel.implemented)


if __name__ == "__main__":
    main()
