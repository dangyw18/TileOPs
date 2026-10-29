# Ascend MoE experts scaffold

This contribution declares interfaces and draft TileLang programs. **No device
computation is implemented.** `compile()` and `--run-kernel` raise
`NotImplementedError` before initializing an NPU or allocating output buffers.
Never enable `implemented` while a draft still contains `T.evaluate(0)`.

## Scope and call graph

```text
FusedMoEFwdOp (existing public composition, target="ascend")
  FusedTopKFwdOp                 <- another contributor; NOT registered here
  FusedMoEExpertsFwdOp
    AscendFusedMoEExpertsKernel  <- whole-Op backend builder, all token counts
      AscendMoEPrePermuteKernel
      AscendMoEGroupedGEMMKernel (gate/up + gated activation)
      AscendMoEGroupedGEMMKernel (down, activation=None)
      AscendMoEPostPermuteKernel (route weights, reduction, scalar, cast)
```

`MoEExpertMLPFwdOp` already composes the two grouped-GEMM leaves. The whole-experts
builder ensures small-T calls also use this scaffold, bypassing CUDA's indexed
small-route specialization. There is no need to add another monolithic
FusedMoEFwd device kernel. Without the colleague's `FusedTopKFwdOp` builder the
full public operator is intentionally unavailable on Ascend.

The four registered public boundaries are PrePermute, GroupedGEMM, PostPermute
and FusedMoEExperts. Shared experts, indexed kernels, quantization, padded/masked
layouts and distributed expert communication are outside this contribution.

## Buffer ABI

| File | Callable input tensors, in order | Output |
|---|---|---|
| pre_permute.py | X[T,H] D, local IDs[T,K] int32 | packed A[P,H] D, ends[E] int32, inverse[P] int32 |
| grouped_gemm.py | A[P,Kdim] D, B[E,N,Kdim] D, ends[E] int32, optional bias[E,N] D | C[P,N] or gated C[P,N/2] |
| post_permute.py | Z[P,H] D, weights[T,K] FP32, inverse[P] int32 | Y[T,H] |
| fused_experts.py | X[T,H], W13[E,2F,H], W2[E,H,F], weights[T,K], IDs[T,K], optional B13[E,2F], B2[E,H] | Y[T,H] |

For pre-permute and the full experts pipeline, P=T*K; standalone post-permute
also accepts an independent materialized P, as its public contract specifies.
D=FP16/BF16. Post-permute output may be FP16 or BF16 independently of its input;
FP32 output is a grouped-GEMM option, not a post-permute option.
Only `ContiguousLayoutSpec.tight_physical_psum()` is declared.
`ends` are physical cumulative ends, NOT per-expert counts; repeated ends mean
empty experts. `inverse[t*K+j]` maps an original route to its packed row. Packing
order within an expert is unspecified; the main checker reconstructs routes and
checks expert segment membership instead of comparing stable-sort identities.

GMM accumulates in FP32, adds optional bias, applies the requested gated
activation, then casts to D or requested FP32. SiLU/GELU read stacked gate||up;
swigluoai reads interleaved gate/up with the existing Op's fixed clipping and
sigmoid constants. Gate/up returns [P,F], so activation is not another public
kernel here. The down output is unweighted. PostPermute computes in FP32:

`Y[t] = cast_D(r * sum_j weights[t,j] * Z[inverse[t*K+j]])`.

The scalar r is applied once, at post-permute. The full existing MoE composition
keeps its TopK sub-op scale at1. Expert down bias is added BEFORE route weighting.
`out=` shape/dtype/device and non-overlap are checked; no input storage is mutated.

## Frontend and backend

All draft builders import `tilelang.ascend.language as T`, open the one-dimensional
`T.Kernel(num_cores)` launch, and compile explicitly with `target="ascend"` and
explicit output buffers. They do not inherit CUDA's Kernel compilation/tuning
machinery. Vector packing/reduction and mixed Cube/Vector GEMM scheduling remain
TODO; the number of cores is a placeholder configuration from Ascend examples,
not an achieved tuning result or a claim about another chip.

`backend.py` uses public `TensorSpec`, `register_detector` and
`register_kernel_builder`. `pyproject.toml` advertises its module under
`tileops.backends`. Importing it performs lazy registration, never compilation.
For a source checkout, explicitly import it before constructing Ops:

```python
from tileops._kernels.moe.ascend import register_backend
from tileops.moe import FusedMoEExpertsFwdOp

register_backend()  # idempotent within a normal process
op = FusedMoEExpertsFwdOp(activation="silu_and_mul", target="ascend")
# op(hidden, w13, w2, weights, ids, out=output)
# Currently raises: the device body is not implemented.
```

The colleague should add only their TopK builder to this same target; do not
register a second detector named ascend. No TopK source is changed here.

MoE kernel re-exports use the package's existing lazy-import pattern so loading
this backend/reference tooling does not load CUDA kernel modules or TileLang.
An explicit `__all__` and the original exported names remain available.

## References and main functions

Each of the four modules has `torch_reference()` and `main()`. These requested
educational oracles are never a runtime fallback. The full experts oracle matches
`workloads/moe.py:ref_routed_experts`: products, activations, biases and accumulation
in FP32, final cast once. Leaf references round at their own output boundary.
When validating the future composed kernel, its stored low-precision leaf
intermediates may differ from the full mathematical oracle; use the existing
workload's verification policy for real regression/benchmark coverage.

Default CLI runs tiny CPU reference inputs and explicitly prints REFERENCE ONLY:

```bash
export PYTHONPATH=/home/d00957057/TileOPs/src
python /home/d00957057/TileOPs/src/tileops/_kernels/moe/ascend/pre_permute.py
python /home/d00957057/TileOPs/src/tileops/_kernels/moe/ascend/grouped_gemm.py --bias
python /home/d00957057/TileOPs/src/tileops/_kernels/moe/ascend/post_permute.py
python /home/d00957057/TileOPs/src/tileops/_kernels/moe/ascend/fused_experts.py --bias
```

After implementing and verifying each device body, opt in with
`--run-kernel --device npu:0`. That path selects the explicit NPU index, constructs
device inputs, invokes the kernel (or public Experts Op), synchronizes, and calls
`torch.testing.assert_close`. No PASS is printed for reference-only runs or
unimplemented kernels. The tiny CLI tolerance is a development smoke check,
not a replacement for the canonical workload verification policy.

Follow `/home/d00957057/tilelang/examples/ascend/example_vecadd.py`,
`example_gemm.py`, `example_mhc_post.py` for real SIMT / Cube / Vector scheduling.
No code in this task rebuilds TileLang or launches an NPU kernel.

## Verification environment

CPU tests are directly runnable, without the repository's device-aware conftest:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=/home/d00957057/TileOPs/src \
  python /home/d00957057/TileOPs/tests/test_moe_ascend_scaffold.py
```

The current NPU environment is Python3.11/Torch2.7; this upstream revision requires
Python>=3.12 and Torch>=2.13,<2.14. Importing public Ops on the old environment
fails on missing `torch.fx.experimental.symbolic_shapes.sym_and`. Do not patch
that dependency or upgrade the existing NPU environment merely to hide the gap.
Backend/reference CPU checks and draft TIR parsing are distinct from unperformed
public-Op integration, vendor compilation, NPU correctness and performance tests.

### Scaffold checks completed

On 2026-10-29: 11 CPU contract tests, eight CPU main variants, four explicit
unimplemented-execution guards, four draft PrimFunc signatures, and all 33 old
MoE exports were checked. Twenty-four small expert-reference cases matched the
canonical workload mathematics. Independent review's PostPermute output-dtype
finding was fixed with a regression test; standalone PostPermute also accepts
independent materialized P. Lint/format and whitespace checks passed. No device
accuracy or performance result is claimed.
