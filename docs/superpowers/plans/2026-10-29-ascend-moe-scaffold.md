# Ascend MoE Scaffold Implementation Plan

> For agentic workers: implement this scoped scaffold inline; no device kernel implementation or NPU execution is requested.

**Goal:** Establish the Ascend experts pipeline on branch `moe` in dangyw18/TileOPs.

**Architecture:** Three independent kernel modules implement the future pre-permute,
M-grouped GEMM, and post-permute boundaries. An experts callable composes them and
registers with the public backend builder protocol, avoiding CUDA's small-route
indexed dispatch. Existing FusedMoEFwdOp remains the full composition; FusedTopK
is a separate colleague's deliverable and is neither implemented nor registered.

**Tech stack:** PyTorch references, TileLang Ascend draft PrimFuncs, TileOPs backend API.

**Spec:** User request: configure origin/upstream, develop on a new MoE branch,
place independent skeletons in src/tileops/_kernels/moe/ascend, include references
and runnable main functions with input generation/call/accuracy checks, using
/home/d00957057/tilelang/examples/ascend as dialect evidence. Do not implement kernels.

## Constraints

- Preserve existing CUDA behavior and public tensor order/dtypes.
- Tight physical prefix-sum layout, non-quantized FP16/BF16; explicitly reject other layouts.
- Routing IDs int32, weights FP32; prefix ends int32; scale applied exactly once.
- No Torch fallback inside kernel execution; every unfinished device kernel raises.
- Each module owns its requested educational torch_reference/main, never used as runtime fallback.
- No TileLang rebuild, package upgrades, or NPU launches in this task.
- Existing remote Python3.11/Torch2.7 cannot import public Ops requiring Torch2.13 sym_and;
  record that limit, do not patch unrelated framework internals to conceal it.

## Tasks

- [x] Add CPU tests first: pack/unpack permutation semantics, empty experts,
  gated GEMM/bias, weighted reduction and exactly-once scaling, unfinished-kernel
  refusal, lazy imports and backend builder tensor/argument contracts.
- [x] Add three draft PrimFuncs and guarded host adapters, independent Torch references,
  small CPU-default main examples with explicit future --run-kernel checks.
- [x] Add experts composition, public backend builders/entry point, and lazy MoE
  kernel exports so importing the Ascend backend does not import CUDA programs.
- [x] Run CPU tests, default mains, syntax/TIR parsing and lint. Inspect final diff;
  document unimplemented kernels, TopK ownership, and public-Op environment limitation.

## Review focus

- Pre-permute row ordering is free within an expert: validate reconstructed routes, not stable-sort identity.
- Empty expert segments and repeated cumulative ends must remain valid.
- Gate/up bias precedes activation; down bias precedes route weighting; no early cast in the mathematical oracle.
- out buffers must preserve shape/dtype/device and may not overlap input memory.
- A draft with T.evaluate(0) is never allowed to compile/execute via the public adapter.
- Full MoE and small-T calls must not silently reach CUDA or implement a second TopK backend.

## Verification and review record

- 11 CPU contract tests pass, including a failing-then-passing regression for
  PostPermute's FP16/BF16 output policy and independent materialized P.
- Eight CPU-default main variants run; four opt-in execution paths explicitly
  refuse unfinished kernels before importing TileLang/torch_npu.
- Four draft PrimFunc signatures parse using the existing TileLang build; no
  vendor compilation or NPU execution was performed.
- Twenty-four small FP16/BF16 reference cases (3 activations, 4 bias-presence
  combinations) match the canonical workload mathematical reference.
- All 33 pre-existing MoE exports still resolve. Ruff lint/format, syntax checks
  and git diff whitespace checks pass.
- Independent read-only review found one PostPermute output-dtype mismatch; fixed
  with the regression above. No remaining actionable findings in scaffold scope.
- Device math/compilation/performance, CUDA device execution and public-Op end-to-end
  integration are unverified by design; the latter also needs the declared Torch2.13
  environment. The colleague still owns TopK. These are recorded scope boundaries,
  not claims of completed numerical kernels.
- Ruling: keep the exact requested checkout and use its clean new branch rather
  than relocating to a worktree; the user explicitly specified this development path.
