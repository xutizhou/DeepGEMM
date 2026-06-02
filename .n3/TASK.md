# DeepGEMM G1/G2 FP4 kernels on DGX Spark

**Current focus**: optimize the DeepGEMM SM121 FP4 1D1D M-grouped kernels used by the G1 and G2 benchmark rows.
**Primary target**: G2 masked M-grouped FP4 (`m_grouped_fp8_fp4_gemm_nt_masked`).
**Secondary target**: G1 contiguous/psum M-grouped FP4 (`m_grouped_fp8_fp4_gemm_nt_contiguous`).
**Machine**: DGX Spark / GB10 / SM121, 48 SMs.
**Starting commit**: `8fbf0ec` on `xutizhou/DeepGEMM`, branch `codex/mgroup-fp4-g1-opt`.

## Context

This is a DeepGEMM kernel optimization task, not a MegaMOE/CuTeDSL task. Use the
MegaMOE task files only as a writing-style reference: clear current focus, hard
constraints, exact benchmark commands, known dead ends, and a workflow that
forces code + correctness + benchmark loops.

The useful implementation area is the shared SM120/SM121 FP8/FP4 1D1D kernel
family. G1 and G2 both exercise that family, so a change must be judged by:

- G2 improvement first.
- G1 non-regression second.
- Correctness on every accepted change.
- Stable repeated measurements, not one good run.

## Target Workloads

### G1: DeepSeek V4 contiguous grouped-M FP4

- API: `deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous`
- groups: `4`
- expected M per group: `8192`
- generated total M: `36096` with fixed seed `124`
- N: `6144`
- K: `7168`
- layout: `NN` through the `nt_contiguous` API
- psum layout: enabled
- A/B: FP4 with UE8M0 scales, `gran_k=32`
- output: BF16
- result log: `avo/results.tsv`, `avo/results.jsonl`

Current GitHub baseline at `8fbf0ec`:

```text
run a: 8290.0 us / 383.5 TFLOP/s / diff 0.01338
run b: 8130.0 us / 391.1 TFLOP/s / diff 0.01338
```

This is about 77-78% of a rough 500 TFLOP/s FP4 peak. G1 target: keep stable
performance near or above this accepted band, and do not accept a G2
improvement that materially regresses G1.

### G2: masked grouped-M FP4

- API: `deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked`
- groups: `6`
- max_m: `4096`
- expected_m: `1024`
- fixed seed: `20260525`
- fixed masked_m: `[744, 940, 722, 747, 1157, 1082]`
- valid_m: `5392`
- N: `4096`
- K: `4096`
- layout: `NN`
- A/B: FP4 with UE8M0 scales, `gran_k=32`
- output: BF16
- compiled dims: `nk`
- result log: `avo/g2_masked_results.tsv`, `avo/g2_masked_results.jsonl`

Historical clean GitHub base at `68e7d2d` before the accepted G2 layout commit:

```text
run a: 618.5 us / 292.5 TFLOP/s / 195.5 GB/s / diff 0.01341
run b: 624.7 us / 289.6 TFLOP/s / 193.5 GB/s / diff 0.01341
run c: 631.1 us / 286.7 TFLOP/s / 191.6 GB/s / diff 0.01341
```

Current GitHub baseline / accepted band at `8fbf0ec`:

```text
run a: 577.3 us / 313.4 TFLOP/s / 209.4 GB/s / diff 0.01341
run b: 592.5 us / 305.4 TFLOP/s / 204.0 GB/s / diff 0.01341
run c: 581.7 us / 311.0 TFLOP/s / 207.8 GB/s / diff 0.01341
```

This is about 62-63% of a rough 500 TFLOP/s FP4 peak. The immediate target is a
stable improvement beyond this accepted band, with the old 10% target versus
the historical clean base still useful as a first milestone: <= 549.6 us, or
equivalently >= about 218.4 GB/s using the benchmark's logical bandwidth field.

## Current Diagnosis

NCU on the accepted G2 BM192/BN128 family shows the remaining gap is mostly
scheduler/barrier/dependency wait, not raw DRAM bandwidth:

- Compute/SOL around 60%.
- Memory/L2 around 47-48%.
- one-or-more eligible around 16%.
- no eligible around 84%.
- active warps/scheduler around 2.25.
- eligible warps/scheduler around 0.21.
- PC sampling dominated by wait/sleep and the consumer full-barrier wait path.
- no local spilling in the accepted lineinfo profile.

Prior accepted shape family:

```text
BM192 / BN128
kNWarps = 4
kTMARegisters = 32
kMMARegisters = 232
scheduler grouping = 7
N-major / multicast-on-A grouping
store sub-tile M = 64
```

## Proposed Approach

Treat this as a kernel-architecture optimization, not a parameter-only sweep.
Each attempt should make one concrete hypothesis about the low-eligible-warp /
barrier-wait profile, then test it.

Priority order:

1. Re-establish the current accepted G2 source at commit `8fbf0ec` and
   benchmark in the current clock/thermal window.
2. Read the accepted SASS/NCU artifacts and map the dominant wait PCs back to
   the source path before editing.
3. Try producer/consumer synchronization changes that shorten the full-barrier
   wait path without adding illegal named-barrier patterns.
4. Try TMA issue ordering or stage-readiness changes only when they preserve the
   single-leader TMA issue shape known to be legal.
5. Try tile/register/scheduler changes only when there is a clear reason tied to
   occupancy, register pressure, or tail waste.
6. After any G2 improvement, run G1 to check for regression.

## Build & Benchmark

Run from the DeepGEMM checkout. On DGX Spark this is normally:

```bash
cd /home/xutingz/gitsrc/DeepGEMM
```

### G1 commands

```bash
# Environment/import sanity
./avo/run_mgroup_moe.sh preflight <iter> <variant>

# Tiny correctness/dispatch smoke
./avo/run_mgroup_moe.sh tiny_smoke <iter> <variant>

# Full G1 benchmark
./avo/run_mgroup_moe.sh mgroup_bench <iter> <variant>
```

### G2 commands

```bash
# Environment/import sanity
python3 avo/bench_g2_masked_fp4.py preflight --iter <iter> --variant <variant>

# Correctness/dispatch smoke
python3 avo/bench_g2_masked_fp4.py tiny --iter <iter> --variant <variant>

# Full G2 benchmark
python3 avo/bench_g2_masked_fp4.py bench --iter <iter> --variant <variant>
```

If the host Python environment cannot import `deep_gemm`, run the same commands
inside the DGX Spark PyTorch container with `PYTHONPATH=/workspace/DeepGEMM`,
matching the prior benchmark setup.

## Key Files

- `csrc/jit_kernels/heuristics/sm120.hpp` — SM120/SM121 tile, pipeline,
  scheduler, and register choices.
- `deep_gemm/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh` — shared
  FP8/FP4 1D1D kernel implementation used by G1/G2.
- `csrc/jit_kernels/heuristics/config.hpp` — launch/config constraints such as
  even `num_sms`.
- `avo/bench_mgroup_moe.py` — G1 correctness and benchmark harness.
- `avo/bench_g2_masked_fp4.py` — G2 correctness and benchmark harness.
- `avo/results.tsv` — G1 log.
- `avo/g2_masked_results.tsv` — G2 log.
- `avo/ncu/` — optional generated NCU reports for accepted and rejected
  variants. If absent in a fresh checkout, capture a new focused profile before
  making profile-driven changes.
- `avo/remote_sass/` — optional generated SASS dumps used for source-level
  diagnosis. If absent, regenerate SASS for the current accepted source.
- `CODEX.md` — optional long optimization history. If absent, rely on this
  task file's dead-end list and benchmark logs.
- `.n3/SM120_SM121_REFERENCES.md` — curated Loom/Blackwell reference map for
  SM120/SM121 DeepGEMM work. Read this before searching Loom manually.
- `/home/xutingz/gitsrc/cuda-knowledge-sm12x/` — same knowledge bundle on DGX
  Spark, with extra public web refs downloaded directly on that machine.

## Key Resources

Use these references the same way the MegaMOE task uses `resources/`: read them
proactively before inventing a new schedule, but validate every idea with the
DeepGEMM G1/G2 harness.

| Resource | Path on DGX Spark | What to look for |
|----------|-------------------|------------------|
| CUDA knowledge audit | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/manifests/reference_quality_audit.md` | Which references are real SM120/SM121 code versus broad background. Start here. |
| CUDA knowledge README | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/README.md` | Directory layout and high-value files for G1/G2. |
| DeepGEMM target mirror | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/DeepGEMM/` | Target SM120 source, PTX wrappers, scheduler, and G1/G2 benchmark files. |
| CUTLASS SM120 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/curated/cutlass/` | Official SM120 grouped NVFP4 examples, builder rules, tile legality, TMA mainloop patterns. |
| FlashInfer SM120 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/curated/flashinfer/` | SM120 grouped FP4/NVFP4 kernels, masked grouped GEMM, fused MoE CuTeDSL patterns. |
| FlashInfer TRTLLM internals | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/flashinfer/` | TensorRT-LLM SM120 heuristic selection, grouped MoE dispatch, TMA warp-specialized traits. |
| TensorRT-LLM SM120 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/TensorRT-LLM/` | SM120 FP4 GEMM template and candidate-config heuristics. |
| SGLang SM120/NVFP4 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/SGLang/` | Practical SM120 NVFP4 wrappers, FlashInfer/TRTLLM MoE integration, DeepGEMM benchmark usage. |
| b12x SM120 CuTeDSL | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/web/b12x/` | Dense NVFP4 GEMM, fused MoE, FP4 quantization, and TMA quantization ideas. |
| flash-attn-4-sm120 | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/web/flash-attn-4-sm120/` | SM120/SM121 CuTeDSL attention constraints: 99KB SMEM, TMA variant, warp specialization. Use as architecture reference, not as a GEMM template. |
| NVIDIA TileGym | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/web/TileGym/` | CUDA Tile group GEMM, MoE, and NVFP4 quantize examples. Broad tile-programming reference, not SM120-only. |
| GitLab FTP mirrors | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/ftp_GitHubSync/` | Internal `/ftp` mirrors for CUTLASS, FlashInfer, TensorRT-LLM, and dynamic-kernel-generator SM120 files. Prefer these when internal GitLab provenance matters. |
| GitLab SM120 repos | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/kaixih_sm120_gemm_fp8/` and `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/petrickl_sm120_mma_benchmarks/` | Direct SM120 standalone GEMM/MMA references found from GitLab. |
| GitLab FP4/NVFP4 repos | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/kaixih_moe-nvfp4/`, `kaixih_nvfp4_cutlass/`, `zhazhang_fp4_kernels/` | FP4/NVFP4 quantization, MoE, and conversion references; useful for scale packing and FP4 handling, less direct for SM120 scheduling. |
| Dynamic Fast Kernel cutlass_ir | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/gitlab/dlarch-fastkernels_cutlass_ir/` | Internal `cutlass_ir` CuTe arch dialect and lowering reference. Current HEAD is SM80/SM100-oriented; use for compiler/lowering context, not as a direct SM120 kernel template. |
| Loom knowledge | `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/curated/loom/` | Blackwell tactics for TMA, mbarrier, warp specialization, scale-factor movement, and DeepGEMM port notes. |

If working locally on the Mac instead of DGX Spark, use the same GitHub
checkout for source edits, but treat the DGX Spark
`/home/xutingz/gitsrc/cuda-knowledge-sm12x/` bundle as the authoritative
reference set for AVO runs.

## Dead Ends — Do Not Retry Without New Evidence

1. 12-warp or 3-warp producer launch shapes that previously hung or failed.
2. Multi-warp TMA issue with named-barrier synchronization; previous attempt
   produced illegal instruction.
3. Partial named-barrier participation in the TMA-store epilogue; previous
   attempts produced illegal instruction.
4. `launch_bounds` min-blocks 2 for this high-register warp-specialized path;
   previous compile/run became non-viable.
5. BM256/BN256, BM96/BN128, BM64/BN128, BM192/BN64, and BM224/BN128 retunes
   unless a new profile explains why the prior rejection no longer applies.
6. Scheduler group sizes 12/15/16 for G2; prior runs regressed.
7. Descending group-order scheduler for the fixed G2 mask; prior run regressed.
8. Tail invalid-subtile skip branches; branch/control perturbation outweighed
   the small saved tail work.
9. Store48 + skip-first TMA-store wait; positive samples were not stable.
10. Treating one good run as real. Any accepted improvement needs repeat
    confirmation.
11. Applying the BM192/BN128 G2 layout globally to G1 psum contiguous. Commit
    `8c4c503` reproduced G2 around 63%, but G1 full shape failed correctness
    with diff about 0.02542. Keep the split path: G1 psum uses legacy
    BM128/BN192 behavior, while G2 masked uses BM192/BN128.

## Acceptance Rules

- Always run correctness before full benchmark.
- Record every attempt with a unique `variant` in the appropriate AVO log.
- Accept only if correctness passes and the improvement is stable across at
  least two benchmark runs.
- For G2, prefer improvements that move toward <= 549.6 us or >= 218.4 GB/s.
- For G1, reject changes that materially regress the accepted band.
- If three consecutive attempts fail, stop editing and inspect NCU/SASS before
  the next code change.
- Restore the last accepted source after every rejected variant.

## Prompt Style For AVO

Use focused prompts. Do not ask for a generic "continue optimizing" loop.

Good prompt shape:

```text
Read .n3/TASK.md and focus only on G2 first.
Re-establish the accepted baseline, inspect the latest NCU/SASS wait path,
then implement one source change aimed at reducing the consumer full-barrier
wait. Run tiny correctness and the full G2 benchmark. If it improves stably,
run G1 to check regression and record the result.
```

For larger investigations, split the work into phases:

1. Read the accepted source and prior NCU report.
2. State one hypothesis and expected signal.
3. Implement one change.
4. Run tiny correctness.
5. Run full benchmark and compare with accepted band.
6. Accept, restore, or gather profile evidence.

## Deep Research / Crew

When stuck, use a small research crew rather than random sweeps. Suggested mix:
`8x nvinf/aws/anthropic/bedrock-claude-opus-4-6 + 4x nvinf/aws/anthropic/bedrock-claude-opus-4-8 + 6x nvinf/azure/openai/gpt-5.5`.

When spawning Crew workers, pass the model explicitly:

- `model="nvinf/aws/anthropic/bedrock-claude-opus-4-6"` for architecture, SASS,
  synchronization, and failure-mode analysis.
- `model="nvinf/aws/anthropic/bedrock-claude-opus-4-8"` for alternate
  architecture hypotheses and second opinions on promising but risky changes.
- `model="nvinf/azure/openai/gpt-5.5"` for broad reference search, synthesis,
  and final experiment planning.

Assign workers to:

- Analyze the accepted G2 NCU/SASS wait PCs and map them to source.
- Search `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/DeepGEMM/`,
  `source_refs/curated/cutlass/`, and `source_refs/curated/loom/` for SM120
  TMA/barrier patterns.
- Search FlashInfer / TensorRT-LLM / SGLang references under
  `/home/xutingz/gitsrc/cuda-knowledge-sm12x/source_refs/` for masked grouped
  GEMM, NVFP4, and legal SM120 heuristic patterns.
- Search `source_refs/web/b12x/` for SM120 CuTeDSL dense GEMM, fused MoE, and
  FP4 quantization ideas that can transfer to the DeepGEMM C++ kernel.
- Search `source_refs/gitlab/ftp_GitHubSync/` for internal GitLab mirror
  versions of CUTLASS, FlashInfer, TensorRT-LLM, and DKG SM120 references.
- Search `source_refs/gitlab/kaixih_sm120_gemm_fp8/` and
  `source_refs/gitlab/petrickl_sm120_mma_benchmarks/` for direct SM120 GEMM/MMA
  examples.
- Search `source_refs/gitlab/dlarch-fastkernels_cutlass_ir/` only for
  `cutlass_ir` / MLIR / CuTe arch lowering questions. Do not use it as a direct
  G2 kernel schedule reference.
- Review rejected attempts in `CODEX.md` if it exists, then summarize what not
  to repeat. If it is absent, use this task file's dead-end list.
- Propose synchronization or scheduling changes that preserve legal SM121
  barrier/TMA patterns.

## Never Stop Criteria

Do not declare "ceiling" without evidence. If no source change works:

1. Re-run the accepted baseline under fixed clock.
2. Capture focused NCU for the current accepted source.
3. Compare SASS/source against the last rejected variant.
4. Update the diagnosis in the log.
5. Pick the next hypothesis from the profile, not from guesswork.

The target is still DeepGEMM G1/G2 performance on DGX Spark.
