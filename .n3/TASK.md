# DeepGEMM DGX Spark G2 FP4 Architecture Optimization Task

Date: 2026-06-15

Primary direction: alternate SM120/SM121 grouped NVFP4 kernel family feasibility.

This task intentionally replaces the previous BK256 producer/TMA issue-order
matrix. Do not continue the old BK256 micro-optimization line unless it is
needed only as the baseline comparator.

## Target

Primary workload:

```text
deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked
G2 masked M-grouped FP4
DGX Spark / GB10 / SM121
```

Current accepted DeepGEMM band to beat:

```text
563-565 us on G2
~320-321 TFLOP/s
~214 GB/s logical bandwidth
diff_vs_ref = 0.01341
```

Historical fast rows around `554-556 us` are not accepted as stable unless they
are reproduced in the same measurement window.

## Why This Direction

The current DeepGEMM BK256 kernel family appears to be at a local ceiling:

- SM busy is about 69%.
- Memory/L2 busy is about 54-55%.
- one-or-more eligible is only about 15-18%.
- achieved occupancy is about 17%.
- registers/thread is about 168.
- dynamic SMEM/block is about 95.49 KiB.
- no spills in the accepted path.

Recent work closed or heavily classified the following directions:

- half-stage / BK128 / BK64 / extra stage depth;
- producer/TMA issue-order atlas;
- A-first, scale-first, and paired scale issue orders;
- BK256 top-family revalidation;
- address/control defuse inside the current source path;
- current-kernel epilogue double-buffering.

The next useful question is whether a different SM120 grouped NVFP4 kernel
family can beat the current DeepGEMM family on the exact G2 workload.

## Required External References

Use local reference code under:

```text
/home/xutingz/gitsrc/cuda-knowledge-sm12x/
```

Priority references:

```text
source_refs/curated/cutlass/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu
source_refs/SGLang/python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_scaled_mm_sm120.cuh
source_refs/SGLang/python/sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh
source_refs/flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
```

Known relevant SM120 tile candidates:

```text
128x128x128
128x128x256
256x128x128
128x256x128
128x128x64
256x128x64
128x256x64
```

Treat SM120 grouped NVFP4 cluster shape `1x1x1` as the default. Do not make
2-CTA cluster/TMEM redesign the first task unless all alternate 1-CTA family
paths are proven infeasible.

## Work Plan

### Phase 0: Baseline and Environment

1. Confirm the current checkout and do not overwrite unrelated user changes.
2. Record exact git status, CUDA/NVIDIA/PyTorch environment, and GPU name.
3. Run or reuse the current DeepGEMM exact-G2 baseline in the same window.
4. Write all artifacts to:

```text
avo/alternate_sm120_grouped_nvfp4/
avo/alternate_sm120_grouped_nvfp4_results.tsv
avo/alternate_sm120_grouped_nvfp4_report.md
```

### Phase 1: Semantic Mapping

Map DeepGEMM G2 inputs to alternate grouped NVFP4 semantics:

- A/B FP4 packing and byte order;
- scale factor layout for SFA/SFB;
- output dtype and epilogue conversion;
- grouped pointer-array format;
- per-group M/N/K and leading dimensions;
- masked valid-M semantics;
- workspace and temporary allocation needs.

If layouts differ, build the smallest adapter or conversion path that lets the
alternate kernel run exact-G2. Measure adapter overhead separately from kernel
time. Do not reject the direction only because the first layout mapping is not
drop-in.

### Phase 2: Standalone Harness

Build a standalone exact-G2 benchmark harness for at least one alternate family.
Preferred order:

1. CUTLASS 79d-style grouped NVFP4.
2. SGLang SM120 NVFP4 grouped / blockwise MoE path.
3. TRT-LLM/FlashInfer CUTLASS heuristic path if available locally.

The harness must support:

- correctness against DeepGEMM/reference output;
- same-window DeepGEMM baseline comparison;
- unique variant names;
- repeated measurements with warmup;
- timing that excludes adapter overhead when reporting kernel-only time;
- a second report line that includes adapter overhead if adapter is needed.

Serialize GPU work:

```bash
flock /tmp/deepgemm_crew_bench.lock <command>
```

### Phase 3: Tile and Schedule Sweep

For each runnable alternate family, sweep supported SM120 tile candidates:

```text
128x128x128
128x128x256
256x128x128
128x256x128
128x128x64
256x128x64
128x256x64
```

For each tile:

1. Build.
2. Run correctness.
3. Capture compile/resource/SASS information if practical.
4. Run same-window benchmark against the current DeepGEMM baseline.
5. Classify as:
   - faster than DeepGEMM;
   - close enough for integration follow-up;
   - slower but diagnostically useful;
   - build-infeasible;
   - semantic mismatch.

### Phase 4: Integration Gate

Only after a standalone alternate kernel is competitive:

- candidate kernel-only median <= current DeepGEMM median, or
- candidate kernel-only median is within about 3% and has a clear path to remove
  adapter overhead or scheduler overhead.

Then propose or implement the smallest DeepGEMM integration path. Do not start
epilogue/TMA-store redesign before this gate.

### Phase 5: Direction 4 Follow-Up

Direction 4 is not independent. It becomes active only after Phase 4 succeeds.

Then evaluate alternate-family epilogue/TMA-store variants:

- epilogue schedule;
- reuse of SMEM C/D;
- delayed TMA store;
- C/D staging;
- adapter removal.

Do not attempt current-DeepGEMM SMEM_D double-buffering again unless a new
SMEM layout removes the known budget blocker.

## Stop Conditions

Do not stop after a single failed build or one failed candidate. This direction
is exhausted only when all of the following are true:

1. CUTLASS-style exact-G2 harness is either benchmarked or proven impossible
   with a concrete semantic/build blocker.
2. SGLang-style SM120 NVFP4 grouped path is either benchmarked or proven
   impossible with a concrete semantic/build blocker.
3. TRT-LLM/FlashInfer heuristic path is checked if local source/build assets are
   available.
4. Supported tile candidates are classified for every runnable family.
5. At least one same-window DeepGEMM baseline/candidate comparison exists for
   each runnable family.
6. The final report explains whether the blocker is layout semantics, build
   support, resource limits, correctness, or performance.

If a competitive candidate is found, do not stop at feasibility. Continue to the
smallest integration plan or patch.

## Rejected Prior Directions

Do not restart these as primary work:

- half-stage/BK128/BK64/stage-depth sweeps;
- simple A/B/SFA/SFB TMA issue-order permutations;
- current-source scale-row/order micro-tweaks;
- scheduler-tail row-major masked-tail small changes;
- current-kernel SMEM_D double-buffering;
- 2-CTA cluster/TMEM redesign before 1-CTA alternate family feasibility.
