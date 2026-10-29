"""Restore routes, multiply FP32 routing coefficients, reduce K, scale once, cast."""

import functools

import torch

from tileops._kernels.moe.ascend.common import (
    KernelScaffold,
    check_dtype,
    check_out,
    check_scale,
    check_tensor,
    example_parser,
    random_tensor,
    run_example,
)

__all__ = ["AscendMoEPostPermuteKernel"]


@functools.lru_cache(maxsize=32)
def _post_permute_program(tokens, top_k, hidden, rows, dtype, out_dtype, scale, num_cores):
    import tilelang.ascend.language as T

    @T.prim_func
    def main(
        Z: T.Tensor((rows, hidden), dtype),
        Weights: T.Tensor((tokens, top_k), "float32"),
        Inverse: T.Tensor((tokens * top_k,), "int32"),
        Y: T.Tensor((tokens, hidden), out_dtype),
    ):
        with T.Kernel(num_cores):
            # TODO: gather Z[Inverse[t*K+j]], multiply Weights[t,j], sum in FP32,
            # multiply scale exactly once, cast/store. No routing/quantization here.
            T.evaluate(0)

    return main


class AscendMoEPostPermuteKernel(KernelScaffold):
    def __init__(
        self,
        num_tokens,
        top_k,
        hidden_size,
        dtype=torch.bfloat16,
        *,
        routed_scaling_factor=1.0,
        out_dtype=None,
        materialized_rows=None,
        num_cores=64,
    ):
        super().__init__(num_cores)
        check_dtype(dtype)
        check_scale(routed_scaling_factor)
        if min(num_tokens, top_k, hidden_size) <= 0:
            raise ValueError("positive dimensions are required")
        if out_dtype not in (None, torch.float16, torch.bfloat16):
            raise ValueError("post-permute out_dtype must be float16 or bfloat16")
        self.num_tokens, self.top_k, self.hidden_size = num_tokens, top_k, hidden_size
        self.materialized_rows = (
            num_tokens * top_k if materialized_rows is None else materialized_rows
        )
        if self.materialized_rows <= 0:
            raise ValueError("materialized_rows must be positive")
        self.dtype, self.out_dtype = dtype, dtype if out_dtype is None else out_dtype
        self.routed_scaling_factor = routed_scaling_factor

    def make_program(self):
        return _post_permute_program(
            self.num_tokens,
            self.top_k,
            self.hidden_size,
            self.materialized_rows,
            str(self.dtype).removeprefix("torch."),
            str(self.out_dtype).removeprefix("torch."),
            self.routed_scaling_factor,
            self.config["num_cores"],
        )

    def __call__(self, expert_output, topk_weights, inverse_indices, *, out=None):
        rows = self.num_tokens * self.top_k
        check_tensor(
            expert_output, "expert_output", (self.materialized_rows, self.hidden_size), self.dtype
        )
        check_tensor(inverse_indices, "inverse_indices", (rows,), torch.int32, expert_output.device)
        check_tensor(
            topk_weights,
            "topk_weights",
            (self.num_tokens, self.top_k),
            torch.float32,
            expert_output.device,
        )
        shape = (self.num_tokens, self.hidden_size)
        check_out(
            out,
            shape,
            self.out_dtype,
            expert_output.device,
            (expert_output, inverse_indices, topk_weights),
        )
        program = self.compile(expert_output.device)
        output = (
            torch.empty(shape, dtype=self.out_dtype, device=expert_output.device)
            if out is None
            else out
        )
        program(expert_output, topk_weights, inverse_indices, output)
        return output


def torch_reference(
    expert_output, topk_weights, inverse_indices, routed_scaling_factor=1.0, *, out_dtype=None
):
    tokens, top_k = topk_weights.shape
    kernel = AscendMoEPostPermuteKernel(
        tokens,
        top_k,
        expert_output.shape[1],
        expert_output.dtype,
        routed_scaling_factor=routed_scaling_factor,
        out_dtype=out_dtype,
        materialized_rows=expert_output.shape[0],
    )
    check_tensor(
        expert_output, "expert_output", (kernel.materialized_rows, kernel.hidden_size), kernel.dtype
    )
    check_tensor(
        inverse_indices, "inverse_indices", (tokens * top_k,), torch.int32, expert_output.device
    )
    check_tensor(topk_weights, "topk_weights", (tokens, top_k), torch.float32, expert_output.device)
    if bool(((inverse_indices < 0) | (inverse_indices >= kernel.materialized_rows)).any()):
        raise ValueError("inverse_indices must be in range")
    restored = (
        expert_output[inverse_indices.long()].float().reshape(tokens, top_k, kernel.hidden_size)
    )
    result = (restored * topk_weights.unsqueeze(-1)).sum(1) * routed_scaling_factor
    return result.to(kernel.out_dtype)


def main():
    args = example_parser(__doc__).parse_args()
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cpu").manual_seed(42)
    z = random_tensor((8, 32), dtype, generator)
    inverse = torch.randperm(8, generator=generator).to(torch.int32)
    weights = torch.randn((4, 2), generator=generator, dtype=torch.float32).softmax(-1)
    kernel = AscendMoEPostPermuteKernel(4, 2, 32, dtype, routed_scaling_factor=2.5)
    expected = torch_reference(z, weights, inverse, 2.5)
    run_example(kernel, (z, weights, inverse), expected, args, implemented=kernel.implemented)


if __name__ == "__main__":
    main()
