"""AVO benchmark+correctness harness for DeepSeek V4 M-grouped FP4.

This is the canonical AVO workload for the DGX Spark MoE optimization task.
It targets the production-relevant contiguous/psum-layout M-grouped path:

    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(...)

The benchmark case matches the DeepSeek V4 SM120/SM121 G1 row:
groups=4, m=36096, n=6144, k=7168, layout=NN, psum_layout=True,
FP4xFP4 1D1D.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import statistics
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_TSV = REPO_ROOT / "avo" / "results.tsv"
RESULTS_JSONL = REPO_ROOT / "avo" / "results.jsonl"
TESTS_DIR = REPO_ROOT / "tests"

MGROUP_NUM_GROUPS = 4
MGROUP_EXPECTED_M = 8192
MGROUP_TARGET_M = 36096
MGROUP_N = 6144
MGROUP_K = 7168
MGROUP_LAYOUT = "NN"
MGROUP_PSUM = True
MGROUP_OUT_DTYPE = "bf16"
MGROUP_DTYPE = "fp4_x_fp4"
MGROUP_SEED = 124

TINY_NUM_GROUPS = 2
TINY_EXPECTED_M = 128
TINY_N = 256
TINY_K = 128

TSV_COLS = [
    "timestamp", "iter", "stage", "variant",
    "m", "n", "k", "layout", "out_dtype", "accumulate", "dtype",
    "median_us", "tflops", "gb_s", "diff_vs_ref",
    "cublas_us", "num_sms", "notes",
]


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _ensure_tsv_header() -> None:
    if RESULTS_TSV.exists() and RESULTS_TSV.stat().st_size > 0:
        return
    RESULTS_TSV.write_text("\t".join(TSV_COLS) + "\n")


def _append_tsv_row(row: dict) -> None:
    _ensure_tsv_header()
    with RESULTS_TSV.open("a") as f:
        f.write("\t".join(str(row.get(c, "")) for c in TSV_COLS) + "\n")


def _append_json_row(row: dict) -> None:
    with RESULTS_JSONL.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _record(stage: str, iteration: int, variant: str, **kw) -> None:
    row = {
        "timestamp": _now(),
        "iter": iteration,
        "stage": stage,
        "variant": variant,
    }
    row.update(kw)
    _append_tsv_row(row)
    _append_json_row(row)


def _seed(seed: int = 0) -> None:
    import random
    import torch  # type: ignore

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _num_sms() -> int:
    try:
        import deep_gemm  # type: ignore

        return int(deep_gemm.get_num_sms())
    except Exception:  # noqa: BLE001
        return -1


def stage_preflight(iteration: int, variant: str) -> dict:
    import torch  # type: ignore
    import deep_gemm  # type: ignore
    from deep_gemm.testing import get_arch_major  # type: ignore

    notes_parts = [
        f"torch={torch.__version__}",
        f"cuda_avail={torch.cuda.is_available()}",
        f"deep_gemm_path={deep_gemm.__path__[0]}",
        f"arch_major={get_arch_major()}",
    ]
    gpu = ""
    cap = ""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        cap_t = torch.cuda.get_device_capability(0)
        cap = f"{cap_t[0]}.{cap_t[1]}"
        notes_parts.extend([f"gpu={gpu}", f"cap={cap}"])
    if hasattr(deep_gemm, "__version__"):
        notes_parts.append(f"deep_gemm_ver={deep_gemm.__version__}")

    notes = ";".join(notes_parts)
    _record("preflight", iteration, variant, num_sms=_num_sms(), notes=notes)
    return {"notes": notes, "gpu": gpu, "cap": cap}


def _setup_alignment() -> int:
    import deep_gemm  # type: ignore

    alignment = int(deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    return alignment


def _build_inputs(num_groups: int, expected_m: int, n: int, k: int):
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))

    import torch  # type: ignore
    from generators import (  # type: ignore
        KernelType, MajorTypeAB, QuantConfig, generate_m_grouped_contiguous,
        get_ue8m0_usage,
    )

    _seed(0)
    _setup_alignment()
    kernel_type = KernelType.Kernel1D1D
    # Match the DeepSeek V4 G1 row: FP4xFP4, layout=NN. The fixed seed makes
    # the generated aligned total M exactly 36096 for expected_m=8192.
    seed = MGROUP_SEED if (num_groups, expected_m, n, k) == (
        MGROUP_NUM_GROUPS, MGROUP_EXPECTED_M, MGROUP_N, MGROUP_K
    ) else 0
    _seed(seed)
    quant_config = QuantConfig((32, 32, True, True))
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    recipe, recipe_a, recipe_b = quant_config.get_recipes()
    m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(
        num_groups, expected_m, n, k,
        MajorTypeAB.KMajor, MajorTypeAB.MNMajor,
        use_ue8m0=use_ue8m0,
        use_psum_layout=MGROUP_PSUM,
        quant_config=quant_config,
    )
    if (num_groups, expected_m, n, k) == (
        MGROUP_NUM_GROUPS, MGROUP_EXPECTED_M, MGROUP_N, MGROUP_K
    ) and m != MGROUP_TARGET_M:
        raise RuntimeError(f"wrong DeepSeek V4 G1 M: got {m}, expected {MGROUP_TARGET_M}")
    return m, a, b, grouped_layout, d, ref_d, recipe, recipe_a, recipe_b, use_ue8m0, quant_config


def _run_kernel(a, b, d, grouped_layout, use_ue8m0, recipe, recipe_a, recipe_b) -> None:
    import deep_gemm  # type: ignore

    expected_m_for_psum_layout = None
    if MGROUP_PSUM and os.environ.get("DG_PSUM_EXPECTED_M"):
        expected_m_for_psum_layout = int(os.environ["DG_PSUM_EXPECTED_M"])

    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        a, b, d, grouped_layout,
        disable_ue8m0_cast=not use_ue8m0,
        use_psum_layout=MGROUP_PSUM,
        expected_m_for_psum_layout=expected_m_for_psum_layout,
        recipe=recipe,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
    )


def _check_diff(d, ref_d, grouped_layout, num_groups: int, quant_config) -> float:
    from deep_gemm.testing import calc_diff  # type: ignore
    from deep_gemm.utils import align, get_mk_alignment_for_contiguous_layout  # type: ignore

    max_diff = 0.0
    for j in range(num_groups):
        start = 0 if j == 0 else align(grouped_layout[j - 1], get_mk_alignment_for_contiguous_layout())
        end = grouped_layout[j]
        max_diff = max(max_diff, float(calc_diff(d[start:end], ref_d[start:end])))
    if max_diff >= quant_config.max_diff():
        raise RuntimeError(f"correctness diff too high: {max_diff:.6f} >= {quant_config.max_diff():.6f}")
    return max_diff


def stage_tiny_smoke(iteration: int, variant: str) -> dict:
    import torch  # type: ignore

    m, a, b, grouped_layout, d, ref_d, recipe, recipe_a, recipe_b, use_ue8m0, qc = _build_inputs(
        TINY_NUM_GROUPS, TINY_EXPECTED_M, TINY_N, TINY_K,
    )
    _run_kernel(a, b, d, grouped_layout, use_ue8m0, recipe, recipe_a, recipe_b)
    torch.cuda.synchronize()
    diff = _check_diff(d, ref_d, grouped_layout, TINY_NUM_GROUPS, qc)
    _record(
        "tiny_smoke", iteration, variant,
        m=m, n=TINY_N, k=TINY_K, layout=MGROUP_LAYOUT,
        out_dtype=MGROUP_OUT_DTYPE, accumulate=0, dtype=MGROUP_DTYPE,
        diff_vs_ref=f"{diff:.5f}", num_sms=_num_sms(),
        notes=f"dispatch_ok;groups={TINY_NUM_GROUPS};expected_m={TINY_EXPECTED_M};psum={int(MGROUP_PSUM)}",
    )
    return {"diff": diff, "m": m}


def stage_mgroup_bench(iteration: int, variant: str, num_repeats: int = 3) -> dict:
    import torch  # type: ignore
    from deep_gemm.testing import bench_kineto, count_bytes  # type: ignore

    m, a, b, grouped_layout, d, ref_d, recipe, recipe_a, recipe_b, use_ue8m0, qc = _build_inputs(
        MGROUP_NUM_GROUPS, MGROUP_EXPECTED_M, MGROUP_N, MGROUP_K,
    )

    _run_kernel(a, b, d, grouped_layout, use_ue8m0, recipe, recipe_a, recipe_b)
    torch.cuda.synchronize()
    diff = _check_diff(d, ref_d, grouped_layout, MGROUP_NUM_GROUPS, qc)

    def _run() -> None:
        _run_kernel(a, b, d, grouped_layout, use_ue8m0, recipe, recipe_a, recipe_b)

    repeats = [bench_kineto(_run, "gemm_", suppress_kineto_output=True) for _ in range(num_repeats)]
    t = float(statistics.median(repeats))
    tflops = 2 * m * MGROUP_N * MGROUP_K / t / 1e12
    gb_s = count_bytes(a, b, d) / 1e9 / t
    alignment = _setup_alignment()
    notes = (
        f"ok;groups={MGROUP_NUM_GROUPS};expected_m={MGROUP_EXPECTED_M};"
        f"target_m={MGROUP_TARGET_M};psum={int(MGROUP_PSUM)};"
        f"alignment={alignment};seed={MGROUP_SEED};recipe_a=(1,32);recipe_b=(1,32)"
    )

    _record(
        "mgroup_bench", iteration, variant,
        m=m, n=MGROUP_N, k=MGROUP_K, layout=MGROUP_LAYOUT,
        out_dtype=MGROUP_OUT_DTYPE, accumulate=0, dtype=MGROUP_DTYPE,
        median_us=f"{t * 1e6:.1f}",
        tflops=f"{tflops:.1f}",
        gb_s=f"{gb_s:.1f}",
        diff_vs_ref=f"{diff:.5f}",
        cublas_us="",
        num_sms=_num_sms(),
        notes=notes,
        repeats_us=[round(r * 1e6, 1) for r in repeats],
        min_us=round(min(repeats) * 1e6, 1),
        max_us=round(max(repeats) * 1e6, 1),
        groups=MGROUP_NUM_GROUPS,
        expected_m_per_group=MGROUP_EXPECTED_M,
        psum_layout=MGROUP_PSUM,
    )
    return {"t_us": t * 1e6, "tflops": tflops, "diff": diff, "m": m}


def stage_dense_fp4_bench(iteration: int, variant: str, num_repeats: int = 3) -> dict:
    import torch  # type: ignore
    import deep_gemm  # type: ignore
    from deep_gemm.testing import bench_kineto, calc_diff, count_bytes  # type: ignore

    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from generators import (  # type: ignore
        KernelType, MajorTypeAB, QuantConfig, generate_normal, get_ue8m0_usage,
    )

    _seed(MGROUP_SEED)
    _setup_alignment()
    kernel_type = KernelType.Kernel1D1D
    quant_config = QuantConfig((32, 32, True, True))
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    recipe, recipe_a, recipe_b = quant_config.get_recipes()
    a, b, _, d, ref_d = generate_normal(
        MGROUP_TARGET_M, MGROUP_N, MGROUP_K,
        MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        accumulate=False,
        out_dtype=torch.bfloat16,
        kernel_type=kernel_type,
        use_ue8m0=use_ue8m0,
        quant_config=quant_config,
    )

    def _run() -> None:
        deep_gemm.fp8_fp4_gemm_nt(
            a, b, d,
            disable_ue8m0_cast=not use_ue8m0,
            recipe=recipe,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )

    _run()
    torch.cuda.synchronize()
    diff = float(calc_diff(d, ref_d))
    if diff >= quant_config.max_diff():
        raise RuntimeError(f"correctness diff too high: {diff:.6f} >= {quant_config.max_diff():.6f}")

    repeats = [bench_kineto(_run, "gemm_", suppress_kineto_output=True) for _ in range(num_repeats)]
    t = float(statistics.median(repeats))
    tflops = 2 * MGROUP_TARGET_M * MGROUP_N * MGROUP_K / t / 1e12
    gb_s = count_bytes(a, b, d) / 1e9 / t
    _record(
        "dense_fp4_bench", iteration, variant,
        m=MGROUP_TARGET_M, n=MGROUP_N, k=MGROUP_K, layout=MGROUP_LAYOUT,
        out_dtype=MGROUP_OUT_DTYPE, accumulate=0, dtype=MGROUP_DTYPE,
        median_us=f"{t * 1e6:.1f}",
        tflops=f"{tflops:.1f}",
        gb_s=f"{gb_s:.1f}",
        diff_vs_ref=f"{diff:.5f}",
        cublas_us="",
        num_sms=_num_sms(),
        notes=(
            "ok;dense_same_shape=1;api=fp8_fp4_gemm_nt;"
            "recipe_a=(1,32);recipe_b=(1,32);layout_b=KMajor"
        ),
        repeats_us=[round(r * 1e6, 1) for r in repeats],
        min_us=round(min(repeats) * 1e6, 1),
        max_us=round(max(repeats) * 1e6, 1),
    )
    return {"t_us": t * 1e6, "tflops": tflops, "diff": diff, "m": MGROUP_TARGET_M}


STAGES = {
    "preflight": stage_preflight,
    "tiny_smoke": stage_tiny_smoke,
    "mgroup_bench": stage_mgroup_bench,
    "dense_fp4_bench": stage_dense_fp4_bench,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=list(STAGES.keys()))
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--variant", type=str, default="baseline_mgroup_moe")
    args = parser.parse_args()
    try:
        STAGES[args.stage](args.iter, args.variant)
        return 0
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().replace("\n", " | ").replace("\t", " ")
        if len(tb) > 800:
            tb = tb[:800] + "...<truncated>"
        _record(args.stage, args.iter, args.variant, num_sms=_num_sms(), notes=f"ERROR:{type(exc).__name__}:{tb}")
        print(f"[bench_mgroup_moe] ERROR in stage={args.stage}: {exc}", file=sys.stderr)
        print(tb, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
