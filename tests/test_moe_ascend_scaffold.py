"""CPU-only contract tests; also runnable directly without tests/conftest.py."""

import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch


class AscendMoEScaffoldTests(unittest.TestCase):
    def test_import_does_not_load_tilelang_or_cuda_kernels(self):
        code = (
            "import sys; import tileops._kernels.moe.ascend; "
            "assert 'tilelang' not in sys.modules; "
            "assert 'tileops._kernels.moe.fused_topk' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pack_handles_empty_experts_and_duplicate_routes(self):
        from tileops._kernels.moe.ascend.pre_permute import torch_reference

        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float16)
        ids = torch.tensor([[2, 0], [2, 2]], dtype=torch.int32)
        packed, ends, inverse = torch_reference(x, ids, 4)
        torch.testing.assert_close(ends, torch.tensor([1, 1, 4, 4], dtype=torch.int32))
        torch.testing.assert_close(packed[inverse.long()], x.repeat_interleave(2, dim=0))
        self.assertEqual(inverse.dtype, torch.int32)

    def test_pack_comparison_allows_another_order_within_one_expert(self):
        from tileops._kernels.moe.ascend.pre_permute import assert_equivalent, torch_reference

        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float16)
        ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        reference = torch_reference(x, ids, 2)
        packed, ends, inverse = reference
        reorder = torch.tensor([1, 0, 3, 2])
        old_to_new = torch.argsort(reorder)
        candidate = (packed[reorder], ends, old_to_new[inverse.long()].int())
        assert_equivalent(candidate, reference, ids)
        bad = (candidate[0], ends, torch.zeros_like(inverse))
        with self.assertRaises(AssertionError):
            assert_equivalent(bad, reference, ids)

    def test_grouped_gemm_bias_activation_and_empty_segment(self):
        from tileops._kernels.moe.ascend.grouped_gemm import torch_reference

        a = torch.tensor([[1, 2], [2, 1]], dtype=torch.float16)
        b = torch.tensor([[[1, 0], [0, 1]], [[5, 5], [6, 6]]], dtype=a.dtype)
        ends = torch.tensor([2, 2], dtype=torch.int32)
        bias = torch.tensor([[1, 3], [7, 8]], dtype=a.dtype)
        y = torch_reference(a, b, ends, bias=bias, out_dtype=torch.float32)
        torch.testing.assert_close(y, torch.tensor([[2, 5], [3, 4]], dtype=torch.float32))
        y = torch_reference(a, b, ends, bias=bias, activation="silu_and_mul")
        expected = (
            torch.nn.functional.silu(torch.tensor([[2.0], [3.0]])) * torch.tensor([[5.0], [4.0]])
        ).half()
        torch.testing.assert_close(y, expected)
        with self.assertRaises(ValueError):
            torch_reference(a, b, torch.tensor([2, 1], dtype=torch.int32))

    def test_post_permute_uses_fp32_weights_and_scale_once(self):
        from tileops._kernels.moe.ascend.post_permute import torch_reference

        z = torch.tensor([[4, 8], [1, 2], [3, 6], [2, 4]], dtype=torch.bfloat16)
        inverse = torch.tensor([1, 3, 2, 0], dtype=torch.int32)
        weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float32)
        expected = torch.tensor([[4.375, 8.75], [8.75, 17.5]], dtype=z.dtype)
        torch.testing.assert_close(torch_reference(z, weights, inverse, 2.5), expected)

    def test_post_permute_output_dtype_matches_public_manifest(self):
        from tileops._kernels.moe.ascend import AscendMoEPostPermuteKernel

        with self.assertRaises(ValueError):
            AscendMoEPostPermuteKernel(2, 2, 4, out_dtype=torch.float32)
        for source, target in ((torch.float16, torch.bfloat16), (torch.bfloat16, torch.float16)):
            kernel = AscendMoEPostPermuteKernel(2, 2, 4, source, out_dtype=target)
            self.assertEqual(kernel.out_dtype, target)

    def test_post_permute_accepts_independent_materialized_row_count(self):
        from tileops._kernels.moe.ascend.backend import build_post_permute
        from tileops._kernels.moe.ascend.post_permute import torch_reference
        from tileops.backend import TensorSpec

        z = torch.tensor([[1, 2], [2, 4], [3, 6]], dtype=torch.bfloat16)
        weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float32)
        inverse = torch.tensor([0, 2, 1, 2], dtype=torch.int32)
        actual = torch_reference(z, weights, inverse, out_dtype=torch.float16)
        torch.testing.assert_close(actual, torch.tensor([[2.5, 5], [2.5, 5]], dtype=torch.float16))

        def spec(tensor):
            return TensorSpec(
                device=torch.device("privateuseone:0"),
                dtype=tensor.dtype,
                shape=tuple(tensor.shape),
            )

        kernel = build_post_permute(
            spec(z),
            spec(weights),
            spec(inverse),
            layout=SimpleNamespace(selection_key="tight_physical_psum"),
            out_dtype=torch.float16,
        )
        self.assertEqual(kernel.materialized_rows, 3)

    def test_kernel_refusal_preserves_out_and_checks_aliasing(self):
        from tileops._kernels.moe.ascend import (
            AscendMoEGroupedGEMMKernel,
            AscendMoEPostPermuteKernel,
        )

        storage = torch.full((8, 4), 13.0, dtype=torch.float16)
        a, out = storage[:4], storage[4:]
        b = torch.ones((2, 4, 4), dtype=a.dtype)
        ends = torch.tensor([2, 4], dtype=torch.int32)
        kernel = AscendMoEGroupedGEMMKernel(4, 2, 4, 4, a.dtype)
        with self.assertRaises(NotImplementedError):
            kernel(a, b, ends, out=out)  # disjoint views are valid buffers
        torch.testing.assert_close(out, torch.full_like(out, 13.0))
        with self.assertRaises(ValueError):
            kernel(a, b, ends, out=a)
        post = AscendMoEPostPermuteKernel(2, 2, 4, a.dtype)
        weights = torch.full((2, 2), 0.5)
        inverse = torch.arange(4, dtype=torch.int32)
        with self.assertRaises(NotImplementedError):
            post(a, weights, inverse)

    def test_experts_reference_down_bias_is_weighted_and_scaled(self):
        from tileops._kernels.moe.ascend.fused_experts import torch_reference

        hidden = torch.zeros((1, 2), dtype=torch.float16)
        w13 = torch.zeros((2, 4, 2), dtype=hidden.dtype)
        w2 = torch.zeros((2, 2, 2), dtype=hidden.dtype)
        ids = torch.tensor([[0, 1]], dtype=torch.int32)
        weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
        bias = torch.tensor([[2, 4], [6, 8]], dtype=hidden.dtype)
        actual = torch_reference(
            hidden, w13, w2, weights, ids, b_down=bias, routed_scaling_factor=2.5
        )
        torch.testing.assert_close(actual, torch.tensor([[12.5, 17.5]], dtype=hidden.dtype))

    def test_draft_kernels_refuse_compilation_without_loading_tilelang(self):
        from tileops._kernels.moe.ascend import (
            AscendMoEGroupedGEMMKernel,
            AscendMoEPostPermuteKernel,
            AscendMoEPrePermuteKernel,
        )

        kernels = (
            AscendMoEPrePermuteKernel(2, 4, 2, 3),
            AscendMoEGroupedGEMMKernel(4, 2, 4, 4),
            AscendMoEPostPermuteKernel(2, 2, 4),
        )
        for kernel in kernels:
            with (
                self.subTest(kernel=type(kernel).__name__),
                self.assertRaisesRegex(NotImplementedError, "not implemented"),
            ):
                kernel.compile()

    def test_backend_builders_keep_tensor_order_and_do_not_claim_topk(self):
        from tileops._kernels.moe.ascend.backend import (
            build_experts,
            build_grouped_gemm,
            build_post_permute,
            register_backend,
        )
        from tileops.backend import TensorSpec, registered_targets

        def spec(shape, dtype=torch.bfloat16):
            return TensorSpec(device=torch.device("privateuseone:0"), dtype=dtype, shape=shape)

        layout = SimpleNamespace(selection_key="tight_physical_psum")
        group = build_grouped_gemm(
            spec((6, 8)),
            spec((3, 10, 8)),
            spec((3,), torch.int32),
            None,
            layout=layout,
            activation="silu_and_mul",
            out_dtype=None,
        )
        self.assertEqual(group.output_shape, (6, 5))
        post = build_post_permute(
            spec((6, 8)),
            spec((2, 3), torch.float32),
            spec((6,), torch.int32),
            layout=layout,
            epilogue=SimpleNamespace(routed_scaling_factor=2.5),
            out_dtype=None,
        )
        self.assertEqual(post.routed_scaling_factor, 2.5)
        experts = build_experts(
            spec((2, 8)),
            spec((3, 10, 8)),
            spec((3, 8, 5)),
            spec((2, 3), torch.float32),
            spec((2, 3), torch.int32),
            None,
            None,
            routed_scaling_factor=2.5,
            activation="silu_and_mul",
        )
        self.assertEqual(experts.post_permute.routed_scaling_factor, 2.5)
        register_backend()
        register_backend()
        self.assertIn("ascend", registered_targets("FusedMoEExpertsFwdOp"))
        self.assertNotIn("ascend", registered_targets("FusedTopKFwdOp"))
        with self.assertRaises(NotImplementedError):
            build_grouped_gemm(
                spec((6, 8)),
                spec((3, 10, 8)),
                spec((3,), torch.int32),
                None,
                layout=SimpleNamespace(selection_key="aligned_physical_psum"),
                activation=None,
                out_dtype=None,
            )


if __name__ == "__main__":
    unittest.main()
