# SM120/SM121 Reference Code For DeepGEMM G1/G2

This file is the short reference map AVO should read before proposing changes
to the DeepGEMM G1/G2 FP8xFP4 M-grouped kernels.

Target platform:

```text
DGX Spark / GB10 / SM121 / 48 SMs
DeepGEMM branch: codex/mgroup-fp4-g1-opt
Starting commit: 8c4c503
```

Use the actual DeepGEMM source first, then read the CUDA knowledge bundle on
DGX Spark. Do not rely on Mac-local `.n3/references/` paths during DGX runs.

## Primary DGX Reference Bundle

```text
/home/xutingz/gitsrc/cuda-knowledge-sm12x/
```

Start with:

- `/home/xutingz/gitsrc/cuda-knowledge-sm12x/README.md`
- `/home/xutingz/gitsrc/cuda-knowledge-sm12x/manifests/reference_quality_audit.md`

The audit separates high-confidence SM120/SM121 code from broad background
material. Read it before using Loom or web references.

## Target DeepGEMM Files

- `csrc/jit_kernels/heuristics/sm120.hpp`
  SM120/SM121 tile, register, scheduler, and stage-selection surface.
- `deep_gemm/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh`
  Shared FP8/FP4 1D1D implementation used by both G1 and G2.
- `deep_gemm/include/deep_gemm/common/sm120_utils.cuh`
  SM120 utility helpers and descriptors.
- `deep_gemm/include/deep_gemm/mma/sm120.cuh`
  SM120 MMA helper layer.
- `deep_gemm/include/deep_gemm/ptx/tcgen05.cuh`
  tcgen05 wrappers used by Blackwell kernels.
- `deep_gemm/include/deep_gemm/ptx/tma.cuh`
  TMA wrapper and memory-movement primitives.
- `deep_gemm/include/deep_gemm/common/tma_copy.cuh`
  DeepGEMM TMA copy helpers.
- `deep_gemm/include/deep_gemm/scheduler/gemm.cuh`
  Grouped GEMM scheduler logic for contiguous and masked M-grouped behavior.

## High-Value SM120/SM121 References

Use these DGX paths directly:

| Resource | DGX path | Why it matters |
|---|---|---|
| DeepGEMM mirror | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/DeepGEMM/` | Target SM120 source, PTX wrappers, scheduler, and benchmark harnesses. |
| CUTLASS SM120 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/curated/cutlass/` | Official SM120 grouped NVFP4 examples, builder rules, 1x1x1 cluster constraints, and TMA mainloop patterns. |
| FlashInfer SM120 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/curated/flashinfer/` | SM120 grouped FP4/NVFP4 kernels, masked grouped GEMM, and fused MoE CuTeDSL patterns. |
| FlashInfer TRTLLM internals | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/flashinfer/` | TensorRT-LLM SM120 heuristic selection and grouped MoE dispatch. |
| TensorRT-LLM SM120 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/TensorRT-LLM/` | FP4 GEMM template and candidate config heuristics. |
| SGLang SM120/NVFP4 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/SGLang/` | NVFP4 SM120 wrappers, FlashInfer/TRTLLM MoE integration, and DeepGEMM benchmark references. |
| b12x SM120 CuTeDSL | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/web/b12x/` | Dense NVFP4 GEMM, fused MoE, FP4 quantization, and TMA quantization ideas. |
| flash-attn-4-sm120 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/web/flash-attn-4-sm120/` | SM120/SM121 constraints such as 99KB SMEM budget, TMA variants, and warp specialization. |
| NVIDIA TileGym | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/web/TileGym/` | CUDA Tile grouped GEMM, MoE, and NVFP4 quantize examples. |
| GitLab FTP mirrors | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/ftp_GitHubSync/` | Internal mirrors for CUTLASS, FlashInfer, TensorRT-LLM, and DKG SM120 files. |
| GitLab SM120 GEMM | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/kaixih_sm120_gemm_fp8/` | Direct SM120 standalone GEMM reference. |
| GitLab SM120 MMA | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/petrickl_sm120_mma_benchmarks/` | Direct SM120 MMA microbenchmark reference. |
| GitLab FP4/NVFP4 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/kaixih_moe-nvfp4/`, `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/kaixih_nvfp4_cutlass/`, `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/zhazhang_fp4_kernels/` | FP4/NVFP4 quantization, MoE, and conversion references. |
| cutlass_ir | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/dlarch-fastkernels_cutlass_ir/` | Compiler/lowering context only; not a direct SM120 kernel schedule template. |
| Loom curated tactics | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/curated/loom/` | Blackwell TMA, mbarrier, warp specialization, scale-factor movement, and DeepGEMM port notes. |

## Official And Public References

- CUDA compute capabilities table:
  `official_docs/cuda_compute_capabilities_13.1.1.html` in the DGX bundle.
  Key facts: CC 12.x has 48 resident warps/SM, 1536 resident threads/SM, 24
  resident blocks/SM, and 99KB max shared memory per thread block in the CUDA
  table.
- CUTLASS SM120 docs:
  `source_refs/curated/cutlass/media/docs/cpp/blackwell_functionality.md`
  in the DGX bundle. Use this for SM120 legal cluster shape, schedule, and
  builder constraints.

## Loom Tactics Most Relevant To G2

Use Loom as architecture/tactic reference, not as SM120 legality proof:

- `source_refs/curated/loom/tactics/sync/implicit_pipeline.md`
- `source_refs/curated/loom/tactics/sync/mbarrier_prefire.md`
- `source_refs/curated/loom/tactics/sync/cta_group2_commit.md`
- `source_refs/curated/loom/tactics/mma/scale_factor_copy_sbo.md`
- `source_refs/curated/loom/tactics/pipeline/multi_stage_pipeline.md`
- `source_refs/curated/loom/tactics/pipeline/double_buffer_tmem.md`
- `source_refs/curated/loom/tactics/memory/smem_tma_swizzled_epilogue.md`
- `source_refs/curated/loom/tactics/warp/warp_specialization.md`
- `source_refs/curated/loom/tactics/warp/elect_sync.md`

Before using any Loom idea, map it to a concrete DeepGEMM source change and
validate it with G2 tiny + repeated G2 benchmark. If G2 improves stably, run G1
to check regression.

## Current Performance Anchor

At commit `8c4c503`:

```text
G1: 8132-8244 us, 385.7-391.0 TFLOP/s, diff 0.01338
G2: 574.7-584.8 us, 309.4-314.8 TFLOP/s, diff 0.01341
```

Treat this as the baseline to beat. Do not use the older 68e7d2d clean-base G2
numbers as the current starting point.

## Caveats

- SM100/B200 examples are not automatically valid for SM120/SM121/GB10. Use
  them for invariants and patterns, then check SM120 headers and generated SASS.
- The current G2 bottleneck is low eligible warps and barrier/dependency wait,
  not a simple bandwidth ceiling. Prefer hypotheses that reduce full-barrier
  wait, improve stage readiness, or overlap epilogue/TMA/MMA work.
- Do not repeat prior rejected attempts listed in `.n3/TASK.md`.
