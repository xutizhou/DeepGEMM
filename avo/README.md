# AVO DeepGEMM M-Grouped Workflow

This directory contains the AVO benchmark harness for the DeepSeek V4
M-grouped contiguous FP4 path on Blackwell client/server targets.

Canonical branch:

```bash
codex/mgroup-fp4-g1-opt
```

## AVO Main-Agent Prompt

The DeepGEMM AVO task contract lives in `.n3/TASK.md`. The first prompt fed to
the main AVO agent lives in `.n3/PROMPT.md`.

On DGX Spark, start AVO with:

```bash
cd /home/xutingz/gitsrc/DeepGEMM
./avo/start_deepgemm_avo_with_prompt.sh
```

The script starts interactive `n3` from `/home/xutingz/gitsrc/avo`, points
`--working-dir` at `/home/xutingz/gitsrc/DeepGEMM`, then injects
`.n3/PROMPT.md` into the tmux session with `tmux load-buffer`, `tmux
paste-buffer`, and `tmux send-keys`.

Useful overrides:

```bash
SESSION=avo-deepgemm-g2-rerun \
AVO_MODEL_PRESET=gpt55 \
./avo/start_deepgemm_avo_with_prompt.sh
```

## Model Selection

The launcher defaults to GPT 5.5 through NVInf/OpenCode:

```bash
./avo/start_deepgemm_avo_with_prompt.sh
```

Preset shortcuts:

```bash
# Main agent: GPT 5.5 / Codex-style path through NVInf
AVO_MODEL_PRESET=gpt55 ./avo/start_deepgemm_avo_with_prompt.sh

# Main agent: Claude Opus 4.6 through NVInf
AVO_MODEL_PRESET=claude46 ./avo/start_deepgemm_avo_with_prompt.sh

# Main agent: Claude Opus 4.7 through NVInf
AVO_MODEL_PRESET=claude47 ./avo/start_deepgemm_avo_with_prompt.sh

# Main agent: Claude Opus 4.8 through NVInf
AVO_MODEL_PRESET=claude48 ./avo/start_deepgemm_avo_with_prompt.sh
```

Full model IDs can override the presets:

```bash
AVO_MODEL=nvinf/azure/openai/gpt-5.5 ./avo/start_deepgemm_avo_with_prompt.sh
AVO_MODEL=nvinf/aws/anthropic/bedrock-claude-opus-4-6 ./avo/start_deepgemm_avo_with_prompt.sh
AVO_MODEL=nvinf/aws/anthropic/bedrock-claude-opus-4-8 ./avo/start_deepgemm_avo_with_prompt.sh
```

Current AVO source has built-in shortcuts/capability entries for Claude 4.6 and
4.7. Claude 4.8 works through the full NVInf model ID above and is exposed here
as the DeepGEMM launcher preset `claude48`.

Crew workers can use different models from the main agent. In the prompt, ask
AVO to create workers with explicit model values, for example:

```text
Use CrewTool to create a small research crew. Spawn 4 workers with
model="nvinf/aws/anthropic/bedrock-claude-opus-4-6" for architecture/SASS
analysis, 4 workers with model="nvinf/aws/anthropic/bedrock-claude-opus-4-8"
for alternate architecture hypotheses, and 2 workers with
model="nvinf/azure/openai/gpt-5.5" for broad reference search and synthesis.
```

This is the same pattern as the Mega-MoE task notes in
`/Users/xutingz/Downloads/misc`, but with DeepGEMM-specific model IDs that route
through the DGX Spark NVInf credential in `~/.config/opencode/opencode.json`.

## Target Workload

Run:

```bash
./avo/run_mgroup_moe.sh mgroup_bench 0 baseline_mgroup_moe
```

This targets:

- API: `deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous`
- groups: `4`
- expected M per group: `8192`
- generated total M: `36096`
- N: `6144`
- K: `7168`
- layout: `NN`
- psum layout: enabled
- A/B: FP4 with UE8M0 scales, `gran_k=32`
- output: BF16

Current DGX Spark GitHub baseline at commit `8fbf0ec`:

```text
G1 run a: m=36096, n=6144, k=7168, median=8290.0 us,
throughput=383.5 TFLOP/s, diff_vs_ref=0.01338
G1 run b: median=8130.0 us, throughput=391.1 TFLOP/s
```

The G2 accepted baseline at the same commit is 577.3-592.5 us,
305.4-313.4 TFLOP/s, diff_vs_ref=0.01341.

## DGX Spark / SM121

The current DGX Spark checkout is:

```bash
/home/xutingz/gitsrc/DeepGEMM
```

Run:

```bash
./avo/run_mgroup_moe.sh preflight 0 preflight
./avo/run_mgroup_moe.sh tiny_smoke 0 tiny_smoke
./avo/run_mgroup_moe.sh mgroup_bench 0 baseline_mgroup_moe
```

## RTX PRO 5000 / SM120

Use the same harness, but point it at the local Python environment and local
build:

```bash
VENV_PY="$(command -v python)" \
DSV4_DEEP_GEMM="$PWD" \
DSV4_PYDEPS="$PWD" \
./avo/run_mgroup_moe.sh mgroup_bench 0 sm120_local_mgroup_moe
```

Direct SM120 smoke/perf scripts:

```bash
python tests/sm120/test_dense_fp8.py
python tests/sm120/test_dense_fp4.py
python tests/sm120/test_dense_fp8_fp4_mixed.py
python tests/sm120/test_m_grouped_fp8.py
```

## Logs

Results are appended to:

```text
avo/results.tsv
avo/results.jsonl
```
