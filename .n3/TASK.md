# Task: Optimize DeepGEMM Dense FP8 D1 on DGX Spark

## Goal

Optimize the DeepGEMM dense FP8 GEMM path on DGX Spark / GB10 / SM121.

Primary benchmark case:

- M = 4096
- N = 16384
- K = 7168
- layout = NN
- output = BF16
- accumulate = false
- dtype = FP8 x FP8

Baseline reference from prior DGX Spark runs:

- latency: approximately 5015-5133 us
- throughput: approximately 187-192 TFLOP/s
- cuBLAS reference: approximately 196-197 TFLOP/s

Initial target: reduce median latency by at least 5% versus the verified baseline.
Stretch target: reduce median latency by at least 10%.

## Required Workflow

1. Verify the current baseline before changing code.
2. Make one optimization hypothesis per iteration.
3. Run correctness and benchmark after every code change.
4. Record every attempt in `avo/results.tsv`.
5. Accept a change only if correctness passes and median latency improves by at least 1.5% over the current best across at least two benchmark runs.
6. Revert or abandon regressions; do not stack changes on top of a failed variant.

## Guardrails

- Do not change the benchmark methodology to make numbers look better.
- Do not optimize unrelated kernels unless this D1 case is blocked.
- Do not treat a one-run improvement below 1.5% as real.
- If three consecutive attempts fail, inspect the SM121 dispatch/wrapper path or collect a profile before making another code change.
- Preserve a clean git history with focused commits.

## Current Hypotheses To Inspect First

- Whether SM121 is only using SM120 wrapper/alias headers.
- Whether the dense D1 case needs SM121-specific tile or scheduling parameters.
- Whether launch, occupancy, or wave/tail effects are limiting the 48-SM GB10 path.
