"""M-grouped GEMM draft; one boundary serves gate/up+activation and down."""

import functools

import torch

from tileops._kernels.moe.ascend.common import (
    KernelScaffold,
    check_activation,
    check_dtype,
    check_out,
    check_tensor,
    example_parser,
    gated_activation,
    random_tensor,
    run_example,
)

__all__ = ["AscendMoEGroupedGEMMKernel"]


@functools.lru_cache(maxsize=32)
def _grouped_gemm_program(rows, experts, n, k, dtype, out_dtype, activation, has_bias, num_cores):
    import tilelang.ascend.language as T

    width = n if activation is None else n // 2
    if has_bias:

        @T.prim_func
        def main(
            A: T.Tensor((rows, k), dtype),
            B: T.Tensor((experts, n, k), dtype),
            Ends: T.Tensor((experts,), "int32"),
            Bias: T.Tensor((experts, n), dtype),
            C: T.Tensor((rows, width), out_dtype),
        ):
            with T.Kernel(num_cores):
                # TODO: variable-M expert scheduler, GM->L1, Cube FP32 accumulation,
                # then bias/activation in Vector and cast/store. Handle empty groups/tails.
                T.evaluate(0)

    else:

        @T.prim_func
        def main(
            A: T.Tensor((rows, k), dtype),
            B: T.Tensor((experts, n, k), dtype),
            Ends: T.Tensor((experts,), "int32"),
            C: T.Tensor((rows, width), out_dtype),
        ):
            with T.Kernel(num_cores):
                # TODO: same grouped Cube/Vector schedule without a bias input.
                # Stacked gate||up for SiLU/GELU; interleaved gate/up for swigluoai.
                T.evaluate(0)

    return main


class AscendMoEGroupedGEMMKernel(KernelScaffold):
    def __init__(
        self,
        rows,
        num_experts,
        n,
        k,
        dtype=torch.bfloat16,
        *,
        activation=None,
        out_dtype=None,
        has_bias=False,
        num_cores=32,
    ):
        super().__init__(num_cores)
        check_dtype(dtype)
        check_activation(activation)
        if rows < 0 or min(num_experts, n, k) <= 0 or (activation is not None and n % 2):
            raise ValueError("invalid grouped GEMM dimensions")
        if out_dtype not in (None, dtype, torch.float32):
            raise ValueError("out_dtype must be the operand dtype or float32")
        self.rows, self.num_experts, self.n, self.k = rows, num_experts, n, k
        self.dtype, self.out_dtype = dtype, dtype if out_dtype is None else out_dtype
        self.activation, self.has_bias = activation, has_bias
        self.output_shape = (rows, n if activation is None else n // 2)

    def make_program(self):
        return _grouped_gemm_program(
            self.rows,
            self.num_experts,
            self.n,
            self.k,
            str(self.dtype).removeprefix("torch."),
            str(self.out_dtype).removeprefix("torch."),
            self.activation,
            self.has_bias,
            self.config["num_cores"],
        )

    def __call__(self, a, b, layout_metadata, bias=None, *, out=None):
        check_tensor(a, "a", (self.rows, self.k), self.dtype)
        check_tensor(b, "b", (self.num_experts, self.n, self.k), self.dtype, a.device)
        check_tensor(layout_metadata, "layout_metadata", (self.num_experts,), torch.int32, a.device)
        if (bias is not None) != self.has_bias:
            raise ValueError("bias presence differs from the built specialization")
        if bias is not None:
            check_tensor(bias, "bias", (self.num_experts, self.n), self.dtype, a.device)
        check_out(out, self.output_shape, self.out_dtype, a.device, (a, b, layout_metadata, bias))
        program = self.compile(a.device)
        output = (
            torch.empty(self.output_shape, dtype=self.out_dtype, device=a.device)
            if out is None
            else out
        )
        if bias is None:
            program(a, b, layout_metadata, output)
        else:
            program(a, b, layout_metadata, bias, output)
        return output


def torch_reference(a, b, physical_ends, *, activation=None, bias=None, out_dtype=None):
    """FP32 products/bias/activation, rounded once at this leaf's output."""
    experts, n, k = b.shape
    kernel = AscendMoEGroupedGEMMKernel(
        a.shape[0],
        experts,
        n,
        k,
        a.dtype,
        activation=activation,
        out_dtype=out_dtype,
        has_bias=bias is not None,
    )
    check_tensor(a, "a", (kernel.rows, k), kernel.dtype)
    check_tensor(b, "b", (experts, n, k), kernel.dtype, a.device)
    check_tensor(physical_ends, "physical_ends", (experts,), torch.int32, a.device)
    if bias is not None:
        check_tensor(bias, "bias", (experts, n), a.dtype, a.device)
    ends = physical_ends.tolist()  # oracle only; never used in device dispatch
    if ends[-1] != a.shape[0] or any(e < s for s, e in zip([0] + ends[:-1], ends, strict=True)):
        raise ValueError("physical_ends must be nondecreasing and end at M")
    output = torch.empty(kernel.output_shape, dtype=kernel.out_dtype, device=a.device)
    start = 0
    for expert, end in enumerate(ends):
        value = a[start:end].float() @ b[expert].float().T
        if bias is not None:
            value = value + bias[expert].float()
        output[start:end] = gated_activation(value, activation).to(kernel.out_dtype)
        start = end
    return output


def main():
    parser = example_parser(__doc__)
    parser.add_argument(
        "--activation",
        choices=("none", "silu_and_mul", "gelu_and_mul", "swigluoai"),
        default="silu_and_mul",
    )
    parser.add_argument("--bias", action="store_true")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    activation = None if args.activation == "none" else args.activation
    generator = torch.Generator(device="cpu").manual_seed(42)
    a, b = random_tensor((8, 32), dtype, generator), random_tensor((4, 32, 32), dtype, generator)
    ends = torch.tensor([2, 2, 5, 8], dtype=torch.int32)
    bias = random_tensor((4, 32), dtype, generator) if args.bias else None
    kernel = AscendMoEGroupedGEMMKernel(
        8, 4, 32, 32, dtype, activation=activation, has_bias=args.bias
    )
    expected = torch_reference(a, b, ends, activation=activation, bias=bias)
    run_example(
        kernel, (a, b, ends), expected, args, implemented=kernel.implemented, kwargs={"bias": bias}
    )


if __name__ == "__main__":
    main()
