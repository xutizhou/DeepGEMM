You are optimizing DeepGEMM G1/G2 FP4 M-grouped kernels on DGX Spark
(GB10 / SM121).

## Context

This is a DeepGEMM kernel optimization task, not a MegaMOE task and not a
CuTeDSL task. Read `.n3/TASK.md` first and treat it as the source of truth for
targets, benchmark commands, accepted baselines, dead ends, and acceptance
rules.

Primary target:

- G2 masked M-grouped FP4:
  `deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked`

Secondary target:

- G1 contiguous/psum M-grouped FP4:
  `deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous`

The useful code path is the shared SM120/SM121 FP8/FP4 1D1D kernel family. G2
gets priority, but any accepted G2 improvement must not materially regress G1.
Start from commit `8fbf0ec` on branch `codex/mgroup-fp4-g1-opt` in the
`xutizhou/DeepGEMM` GitHub checkout.

## The Problem

The historical clean GitHub base at `68e7d2d` was:

```text
618.5 / 624.7 / 631.1 us
292.5 / 289.6 / 286.7 TFLOP/s
195.5 / 193.5 / 191.6 GB/s
diff 0.01341
```

The current GitHub baseline / accepted family at `8fbf0ec` is:

```text
577.3 / 592.5 / 581.7 us
313.4 / 305.4 / 311.0 TFLOP/s
209.4 / 204.0 / 207.8 GB/s
diff 0.01341
```

This is about 62-63% of a rough 500 TFLOP/s FP4 peak. The first milestone is
still the old 10% target versus the historical clean base:

```text
<= 549.6 us, or >= about 218.4 GB/s
```

Prior NCU says the remaining bottleneck is mostly scheduler/barrier/dependency
wait, not raw DRAM bandwidth:

- one-or-more eligible around 16%
- no eligible around 84%
- active warps/scheduler around 2.25
- eligible warps/scheduler around 0.21
- PC sampling dominated by wait/sleep and the consumer full-barrier wait path
- no local spilling in the accepted lineinfo profile

## Proposed Approach

Do not run a random parameter sweep. Work from profile evidence.

First re-establish the accepted G2 source at commit `8fbf0ec` and benchmark in
the current clock/thermal window. Then inspect accepted NCU/SASS artifacts if
they exist; if they are absent in a fresh checkout, capture a focused profile
before making profile-driven changes. Map the dominant wait path back to source
before editing. Implement one concrete source change aimed at reducing the
low-eligible-warp / full-barrier-wait issue, then run correctness and
benchmark.

Prefer changes that preserve the known legal shape:

```text
BM192 / BN128
kNWarps = 4
kTMARegisters = 32
kMMARegisters = 232
scheduler grouping = 7
N-major / multicast-on-A grouping
store sub-tile M = 64
```

## Your Task

1. Read `.n3/TASK.md` completely.
2. Read `.n3/SM120_SM121_REFERENCES.md`; use it as the short map into Loom
   reference code and tactics.
3. Read the CUDA knowledge audit and README before inventing a new schedule:
   `/home/xutingz/gitsrc/cuda-knowledge-sm12x/manifests/reference_quality_audit.md`
   and `/home/xutingz/gitsrc/cuda-knowledge-sm12x/README.md`.
4. Explore the required SM120/SM121 reference resources listed below, the same
   way the MegaMOE prompt explores `resources/`.
5. Read the current accepted code path in:
   `deep_gemm/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh`
   and `csrc/jit_kernels/heuristics/sm120.hpp`.
6. Re-establish the G2 accepted baseline with the provided G2 harness.
7. Inspect the latest G2 NCU/SASS artifacts in `avo/ncu/` and
   `avo/remote_sass/` if they exist; otherwise capture a fresh focused profile.
   Map the dominant wait PCs to source.
8. State one hypothesis and expected signal before editing.
9. Implement one scoped kernel or heuristic change.
10. Run G2 tiny correctness first, then full G2 benchmark.
11. If G2 improves stably, run G1 to check regression.
12. Record the result using a unique variant name in the appropriate AVO log.
13. If the variant fails correctness or regresses perf, restore the last
    accepted source before the next attempt.

## Required Knowledge References

Use these references proactively, not as optional reading. Start with the audit
so you do not waste time on weak references.

| Resource | Path on DGX Spark | What to look for |
|----------|-------------------|------------------|
| CUDA knowledge audit | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/manifests/reference_quality_audit.md` | Which references are high-confidence SM120/SM121 source code. |
| DeepGEMM target mirror | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/DeepGEMM/` | Target SM120 implementation, PTX wrappers, scheduler, and benchmark harnesses. |
| CUTLASS SM120 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/curated/cutlass/` | Official grouped NVFP4 example, builder rules, 1x1x1 cluster constraints, TMA mainloop structure. |
| FlashInfer SM120 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/curated/flashinfer/` | SM120 grouped FP4/NVFP4, masked grouped GEMM, fused MoE CuTeDSL reference. |
| FlashInfer TRTLLM internals | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/flashinfer/` | SM120 heuristic selection and MoE grouped GEMM TMA warp-specialized dispatch. |
| TensorRT-LLM SM120 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/TensorRT-LLM/` | FP4 GEMM template and candidate config heuristics. |
| SGLang SM120/NVFP4 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/SGLang/` | NVFP4 SM120 wrappers, FlashInfer/TRTLLM MoE integration, DeepGEMM benchmark references. |
| b12x SM120 CuTeDSL | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/web/b12x/` | Dense NVFP4 GEMM, fused MoE, FP4 quantization, TMA quantization. Use for transferable ideas. |
| flash-attn-4-sm120 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/web/flash-attn-4-sm120/` | SM120/SM121 hardware constraints, 99KB SMEM budgeting, TMA/warp-specialized attention patterns. |
| NVIDIA TileGym | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/web/TileGym/` | CUDA Tile group GEMM, MoE, and NVFP4 quantize patterns. |
| GitLab FTP mirrors | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/ftp_GitHubSync/` | Internal `/ftp` mirrors for CUTLASS, FlashInfer, TensorRT-LLM, and DKG SM120 files. |
| GitLab SM120 repos | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/kaixih_sm120_gemm_fp8/` and `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/petrickl_sm120_mma_benchmarks/` | Direct SM120 standalone GEMM/MMA examples. |
| GitLab FP4/NVFP4 repos | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/kaixih_moe-nvfp4/`, `kaixih_nvfp4_cutlass/`, `zhazhang_fp4_kernels/` | FP4/NVFP4 quantization, MoE, and conversion references. |
| Dynamic Fast Kernel cutlass_ir | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/dlarch-fastkernels_cutlass_ir/` | Internal `cutlass_ir` CuTe arch dialect and lowering reference; current HEAD is not direct SM120 kernel code. |
| Loom curated tactics | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/curated/loom/` | Blackwell TMA, mbarrier, warp specialization, FP4 scale-factor tactics. |

## Key Files

- `csrc/jit_kernels/heuristics/sm120.hpp` - SM120/SM121 tile, pipeline,
  scheduler, and register choices.
- `deep_gemm/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh` - shared
  FP8/FP4 1D1D kernel used by G1/G2.
- `csrc/jit_kernels/heuristics/config.hpp` - launch/config constraints.
- `avo/bench_g2_masked_fp4.py` - G2 correctness and benchmark harness.
- `avo/bench_mgroup_moe.py` - G1 correctness and benchmark harness.
- `avo/g2_masked_results.tsv` - G2 result log.
- `avo/results.tsv` - G1 result log.
- `avo/ncu/` - optional generated G2 NCU reports.
- `avo/remote_sass/` - optional generated SASS dumps.
- `CODEX.md` - optional long optimization history; if absent, rely on
  `.n3/TASK.md` dead ends and benchmark logs.
- `.n3/SM120_SM121_REFERENCES.md` - curated Loom/Blackwell reference map for
  SM120/SM121 DeepGEMM work.
- `/home/xutingz/gitsrc/cuda-knowledge-sm12x/` - DGX Spark CUDA knowledge
  bundle with DeepGEMM, CUTLASS, FlashInfer, TensorRT-LLM, SGLang, b12x,
  flash-attn-4-sm120, TileGym, GitLab `/ftp` mirrors, GitLab SM120 repos, Loom,
  and official docs.

## Deep Research

Use CrewTool when stuck, but make the research specific. Suggested mix:
`8x nvinf/aws/anthropic/bedrock-claude-opus-4-6 + 4x nvinf/aws/anthropic/bedrock-claude-opus-4-8 + 6x nvinf/azure/openai/gpt-5.5`.

When spawning Crew workers, pass the model explicitly:

- `model="nvinf/aws/anthropic/bedrock-claude-opus-4-6"` for architecture, SASS,
  synchronization, and failure-mode analysis.
- `model="nvinf/aws/anthropic/bedrock-claude-opus-4-8"` for alternate
  architecture hypotheses and second opinions on promising but risky changes.
- `model="nvinf/azure/openai/gpt-5.5"` for broad reference search, synthesis,
  and final experiment planning.

Focus workers on:

- Mapping accepted G2 NCU/SASS wait PCs to the source.
- Searching `source_refs/DeepGEMM/`, `source_refs/curated/cutlass/`, and
  `source_refs/curated/loom/` under
  `/home/xutingz/gitsrc/cuda-knowledge-sm12x/` for SM120/SM121 TMA/barrier
  scheduling patterns.
- Searching FlashInfer, TensorRT-LLM, SGLang, and b12x references in the same
  knowledge bundle for masked grouped GEMM, NVFP4, and legal SM120 heuristic
  patterns.
- Searching `source_refs/gitlab/ftp_GitHubSync/` for internal GitLab mirror
  versions of CUTLASS, FlashInfer, TensorRT-LLM, and DKG SM120 references.
- Searching `source_refs/gitlab/kaixih_sm120_gemm_fp8/` and
  `source_refs/gitlab/petrickl_sm120_mma_benchmarks/` for direct SM120 GEMM/MMA
  standalone examples.
- Searching `source_refs/gitlab/dlarch-fastkernels_cutlass_ir/` only for
  `cutlass_ir` / MLIR / CuTe arch lowering questions.
- Summarizing rejected attempts from `CODEX.md` if present so they are not
  repeated. If absent, use `.n3/TASK.md` dead ends.
- Proposing one legal synchronization or scheduling change that can be tested
  without changing benchmark methodology.

## Dead Ends - Do Not Retry Without New Evidence

1. 12-warp or 3-warp producer launch shapes that previously hung or failed.
2. Multi-warp TMA issue with named-barrier synchronization; previous attempt
   produced illegal instruction.
3. Partial named-barrier participation in the TMA-store epilogue; previous
   attempts produced illegal instruction.
4. `launch_bounds` min-blocks 2 for this high-register warp-specialized path.
5. BM256/BN256, BM96/BN128, BM64/BN128, BM192/BN64, and BM224/BN128 retunes
   unless a new profile explains why the prior rejection no longer applies.
6. Scheduler group sizes 12/15/16 for G2; prior runs regressed.
7. Descending group-order scheduler for the fixed G2 mask; prior run regressed.
8. Tail invalid-subtile skip branches; prior runs regressed.
9. Store48 + skip-first TMA-store wait; positive samples were not stable.
10. Treating one good run as real.
11. Applying the BM192/BN128 G2 layout globally. It reproduced G2, but broke G1
    psum contiguous correctness. Keep G1/G2 split by `GemmType`.

## Constraints

- Optimize G2 first; check G1 only after a real G2 candidate.
- Do not optimize dense G2, D1, K-grouped, or unrelated kernels.
- Do not change benchmark methodology to create an artificial speedup.
- Do not accept a result without correctness.
- Do not accept a result from one good run.
- Do not declare the target impossible without fresh profile evidence.
- Do not read or write unrelated task files outside this DeepGEMM checkout.

## Git

Use focused commits only after a working, verified change. Do not push to main.
If the working tree contains unrelated user changes, leave them alone.

## Build & Test

Run from the DeepGEMM checkout:

```bash
cd /home/xutingz/gitsrc/DeepGEMM
```

G2:

```bash
python3 avo/bench_g2_masked_fp4.py preflight --iter <iter> --variant <variant>
python3 avo/bench_g2_masked_fp4.py tiny --iter <iter> --variant <variant>
python3 avo/bench_g2_masked_fp4.py bench --iter <iter> --variant <variant>
```

G1:

```bash
./avo/run_mgroup_moe.sh preflight <iter> <variant>
./avo/run_mgroup_moe.sh tiny_smoke <iter> <variant>
./avo/run_mgroup_moe.sh mgroup_bench <iter> <variant>
```

If host Python cannot import `deep_gemm`, run inside the DGX Spark PyTorch
container with `PYTHONPATH=/workspace/DeepGEMM`, matching prior benchmark setup.

## Workflow

Read `.n3/TASK.md` -> read `.n3/SM120_SM121_REFERENCES.md` -> read the
`cuda-knowledge-sm12x` audit/README -> inspect high-value reference code ->
re-establish G2 baseline -> inspect NCU/SASS -> state one hypothesis ->
implement one change -> G2 tiny -> G2 bench -> repeat G2 bench if positive ->
G1 regression check -> commit or restore -> iterate.

Spend enough time reading to avoid repeating known dead ends, but do not spend
more than 1 hour reading without starting the first concrete experiment.
