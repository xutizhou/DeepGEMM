You are optimizing DeepGEMM G2 FP4 M-grouped performance on DGX Spark
(GB10 / SM121). Read `.n3/TASK.md` and
`.n3/ALTERNATE_SM120_GROUPED_NVFP4_CONFIG.md` first. Treat them as the source
of truth.

## Immediate Execution Requirement

Do not stop after planning or context inventory. After reading the two required
files, immediately do these actions:

1. Create or update `avo/alternate_sm120_grouped_nvfp4_report.md` with current
   status.
2. Run the exact-G2 DeepGEMM baseline under:

```bash
flock /tmp/deepgemm_crew_bench.lock <command>
```

using `avo/bench_g2_masked_fp4.py preflight`, then `tiny`, then `bench`, with a
unique variant name beginning:

```text
alternate_sm120_grouped_nvfp4_deepgemm_baseline_
```

3. After the baseline, begin CUTLASS 79d semantic/build feasibility.

Do not ask for confirmation before these actions.

## Objective

Make direction #1 the main optimization direction:

```text
alternate SM120/SM121 grouped NVFP4 kernel family feasibility
```

This is not the previous BK256 producer/TMA issue-order matrix. The old BK256
path is only the baseline comparator.

## Required Behavior

Work in CREW mode if available. Split the work into owners such as:

- reference-mapping owner;
- exact-G2 harness owner;
- tile-sweep/benchmark owner;
- correctness/layout owner;
- integration-gate owner.

Do not stop after one failed candidate. This direction is exhausted only after
CUTLASS-style, SGLang-style, and locally available TRT-LLM/FlashInfer-style
SM120 grouped NVFP4 paths have been classified with concrete evidence.

## Immediate First Steps

1. Read `.n3/TASK.md`.
2. Read `.n3/ALTERNATE_SM120_GROUPED_NVFP4_CONFIG.md`.
3. Read `avo/ARCHITECTURE_FEASIBILITY_NOTES.md` if present.
4. Read the local references under `/home/xutingz/gitsrc/cuda-knowledge-sm12x/`
   listed in the task file.
5. Confirm current DeepGEMM exact-G2 baseline band in the same window.
6. Build the smallest standalone exact-G2 alternate-kernel benchmark harness.

## Target Comparator

Current DeepGEMM accepted band:

```text
G2: 563-565 us, diff_vs_ref=0.01341
```

Any alternate candidate must be compared against an adjacent DeepGEMM baseline,
not against stale historical rows.

## Measurement Discipline

Use unique variant names. Serialize GPU work:

```bash
flock /tmp/deepgemm_crew_bench.lock <command>
```

Report kernel-only timing separately from adapter/conversion overhead. If a
layout adapter is needed, do not hide its cost.

Use warmup and repeated measurements. Prefer same-window baseline/candidate
pairs. Record slow tails, not just best rows.

## Artifacts

Write results here:

```text
avo/alternate_sm120_grouped_nvfp4/
avo/alternate_sm120_grouped_nvfp4_results.tsv
avo/alternate_sm120_grouped_nvfp4_report.md
```

The report must include:

- exact references used;
- semantic mapping status;
- build status for each family;
- tile candidates tested;
- correctness status;
- same-window timing;
- whether a candidate is worth integration;
- if not, the precise reason this architecture direction is exhausted.

## Hard Non-Goals

Do not restart:

- half-stage/BK128/BK64/stage-depth sweeps;
- simple PWG issue-order atlas work;
- current-source scale-order micro-tweaks;
- current-kernel SMEM_D double-buffering;
- 2-CTA cluster/TMEM rewrite before 1-CTA alternate-family feasibility.

If a competitive alternate family is found, continue to the smallest integration
plan or patch. Direction #4, epilogue/TMA-store redesign, is only allowed after
the alternate family is competitive.
