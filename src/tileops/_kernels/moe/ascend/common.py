"""Host-only helpers for the deliberately unimplemented Ascend MoE kernels."""

from __future__ import annotations

import argparse
import math
from typing import ClassVar

import torch

from tileops._kernels.buffer_utils import tensors_overlap

__all__ = ["KernelScaffold"]


def check_tensor(tensor, name, shape, dtype, device=None):
    """Validate the static tensor ABI without reading device values."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}")


def check_dtype(dtype):
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Ascend MoE operands must be float16 or bfloat16")


def check_scale(scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("routed_scaling_factor must be finite and positive")


def check_activation(activation):
    if activation not in (None, "silu_and_mul", "gelu_and_mul", "swigluoai"):
        raise ValueError(f"unknown gated activation {activation!r}")


def gated_activation(value, activation):
    """Mathematical oracle only, matching MoEGroupedGEMMFwdOp's row order."""
    check_activation(activation)
    if activation is None:
        return value
    if value.shape[-1] % 2:
        raise ValueError("gated activation requires an even projection width")
    if activation == "swigluoai":
        gate = value[..., 0::2].clamp(max=7.0)
        up = value[..., 1::2].clamp(-7.0, 7.0)
        return gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
    gate, up = value.chunk(2, dim=-1)
    if activation == "silu_and_mul":
        return torch.nn.functional.silu(gate) * up
    return torch.nn.functional.gelu(gate, approximate="none") * up


def check_out(out, shape, dtype, device, inputs):
    if out is None:
        return
    check_tensor(out, "out", shape, dtype, device)
    if any(t is not None and tensors_overlap(out, t) for t in inputs):
        raise ValueError("out must not overlap an input")


class KernelScaffold:
    """Guard draft TIR from compilation and execution until its body is implemented."""

    implemented: ClassVar[bool] = False

    def __init__(self, num_cores):
        if num_cores <= 0:
            raise ValueError("num_cores must be positive")
        self.config = {"num_cores": num_cores}
        self._compiled = {}

    def compile(self, device=None):
        # FIXME(staged-rollout): The requested contribution is a framework, not device math.
        # Broken invariant: draft PrimFuncs have no computation and cannot produce results.
        # Why: device implementation is intentionally deferred to the next development step.
        # Cleanup: implement/verify make_program() before setting implemented=True.
        if not self.implemented:
            raise NotImplementedError(
                f"{type(self).__name__}: Ascend device body is not implemented"
            )
        if device is None or device.type not in ("npu", "privateuseone"):
            raise ValueError("an explicit NPU device is required")
        if device not in self._compiled:
            import tilelang

            # Explicit outputs preserve the public out-buffer ABI. No CUDA Kernel base,
            # CUDA tuning/SM queries or default target are involved.
            with torch.npu.device(device):
                self._compiled[device] = tilelang.compile(
                    self.make_program(), target="ascend", out_idx=None
                )
        return self._compiled[device]

    def request_tune(self):
        raise NotImplementedError("Ascend MoE tuning is not implemented")


def example_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="npu:0", help="explicit NPU index for --run-kernel")
    parser.add_argument(
        "--run-kernel",
        action="store_true",
        help="compare the implemented kernel with the CPU oracle",
    )
    return parser


def random_tensor(shape, dtype, generator):
    return (torch.randn(shape, generator=generator, device="cpu") * 0.1).to(dtype)


def run_example(kernel, inputs, expected, args, *, implemented, kwargs=None, compare=None):
    """CPU by default; no unfinished kernel can report a successful comparison."""
    print("REFERENCE ONLY: CPU oracle computed; no device kernel comparison has run.")
    if not args.run_kernel:
        return
    if not implemented:
        raise NotImplementedError("Ascend device body is not implemented; no NPU was initialized")
    import torch_npu  # noqa: F401

    device = torch.device(args.device)
    if device.type != "npu" or device.index is None:
        raise ValueError("--device must be an explicit npu:<index>")
    torch.npu.set_device(device)

    def move(value):
        return value.to(device) if isinstance(value, torch.Tensor) else value

    actual = kernel(*(move(t) for t in inputs), **{k: move(v) for k, v in (kwargs or {}).items()})
    torch.npu.synchronize(device)
    if compare is None:
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)
    else:
        compare(actual, expected)
    print("PASS: Ascend kernel output compared with torch_reference")
