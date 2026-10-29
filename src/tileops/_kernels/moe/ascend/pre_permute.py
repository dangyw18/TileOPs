"""Tight expert packing: X[T,H], IDs[T,K] -> A[T*K,H], ends[E], inverse[T*K]."""

import functools

import torch

from tileops._kernels.moe.ascend.common import (
    KernelScaffold,
    check_dtype,
    check_tensor,
    example_parser,
    random_tensor,
    run_example,
)

__all__ = ["AscendMoEPrePermuteKernel"]


@functools.lru_cache(maxsize=32)
def _pre_permute_program(tokens, hidden, top_k, experts, dtype, num_cores):
    import tilelang.ascend.language as T

    rows = tokens * top_k

    @T.prim_func
    def main(
        X: T.Tensor((tokens, hidden), dtype),
        IDs: T.Tensor((tokens, top_k), "int32"),
        A: T.Tensor((rows, hidden), dtype),
        Ends: T.Tensor((experts,), "int32"),
        Inverse: T.Tensor((rows,), "int32"),
    ):
        with T.Kernel(num_cores):
            # TODO: histogram / prefix sum / scatter, including empty experts.
            # These may require several launches; do not assume a cross-core barrier.
            # Inverse[t*K+j] must name the packed row for this original route.
            T.evaluate(0)

    return main


class AscendMoEPrePermuteKernel(KernelScaffold):
    """Host ABI for tight physical_psum packing; draft device code is never executable."""

    def __init__(
        self, num_tokens, hidden_size, top_k, num_experts, dtype=torch.bfloat16, num_cores=64
    ):
        super().__init__(num_cores)
        check_dtype(dtype)
        if min(num_tokens, hidden_size, top_k, num_experts) <= 0 or num_tokens * top_k >= 2**31:
            raise ValueError("positive dimensions and fewer than 2**31 routes are required")
        self.num_tokens, self.hidden_size = num_tokens, hidden_size
        self.top_k, self.num_experts, self.dtype = top_k, num_experts, dtype

    def make_program(self):
        return _pre_permute_program(
            self.num_tokens,
            self.hidden_size,
            self.top_k,
            self.num_experts,
            str(self.dtype).removeprefix("torch."),
            self.config["num_cores"],
        )

    def __call__(self, hidden_states, local_expert_ids):
        check_tensor(
            hidden_states, "hidden_states", (self.num_tokens, self.hidden_size), self.dtype
        )
        check_tensor(
            local_expert_ids,
            "local_expert_ids",
            (self.num_tokens, self.top_k),
            torch.int32,
            hidden_states.device,
        )
        program = self.compile(hidden_states.device)  # raises before output allocation today
        rows = self.num_tokens * self.top_k
        packed = hidden_states.new_empty((rows, self.hidden_size))
        ends = torch.empty((self.num_experts,), dtype=torch.int32, device=hidden_states.device)
        inverse = torch.empty((rows,), dtype=torch.int32, device=hidden_states.device)
        program(hidden_states, local_expert_ids, packed, ends, inverse)
        return packed, ends, inverse


def torch_reference(hidden_states, local_expert_ids, num_experts):
    """Stable oracle; production kernels may choose any order within one expert."""
    tokens, hidden = hidden_states.shape
    top_k = local_expert_ids.shape[1]
    AscendMoEPrePermuteKernel(tokens, hidden, top_k, num_experts, hidden_states.dtype)
    check_tensor(
        local_expert_ids, "local_expert_ids", (tokens, top_k), torch.int32, hidden_states.device
    )
    ids = local_expert_ids.flatten().long()
    if bool(((ids < 0) | (ids >= num_experts)).any()):
        raise ValueError("expert IDs must be local and in range")
    order = torch.argsort(ids, stable=True)
    packed = hidden_states.index_select(0, order // top_k)
    ends = torch.bincount(ids, minlength=num_experts).cumsum(0).to(torch.int32)
    inverse = torch.empty_like(order, dtype=torch.int32)
    inverse[order] = torch.arange(order.numel(), dtype=torch.int32, device=order.device)
    return packed, ends, inverse


def assert_equivalent(actual, reference, local_expert_ids):
    """Check semantic packing, not the oracle's arbitrary stable-sort order."""
    packed, ends, inverse = (t.cpu() for t in actual)
    ref_packed, ref_ends, ref_inverse = (t.cpu() for t in reference)
    torch.testing.assert_close(ends, ref_ends, rtol=0, atol=0)
    torch.testing.assert_close(
        inverse.sort().values, torch.arange(inverse.numel(), dtype=torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        packed[inverse.long()], ref_packed[ref_inverse.long()], rtol=0, atol=0
    )
    ids = local_expert_ids.cpu().flatten().long()
    starts = torch.cat((torch.zeros(1, dtype=torch.int32), ends[:-1]))
    assert bool(((inverse >= starts[ids]) & (inverse < ends[ids])).all())


def main():
    args = example_parser(__doc__).parse_args()
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cpu").manual_seed(42)
    x = random_tensor((4, 32), dtype, generator)
    ids = torch.tensor([[2, 0], [2, 2], [0, 2], [2, 0]], dtype=torch.int32)
    kernel = AscendMoEPrePermuteKernel(4, 32, 2, 4, dtype)
    expected = torch_reference(x, ids, 4)
    run_example(
        kernel,
        (x, ids),
        expected,
        args,
        implemented=kernel.implemented,
        compare=lambda actual, ref: assert_equivalent(actual, ref, ids),
    )


if __name__ == "__main__":
    main()
