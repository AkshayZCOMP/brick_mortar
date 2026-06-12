"""Compare aligned vs staggered gap placement for QI layups in 0 deg and 90 deg tension.

Each layup is evaluated under both in-plane tension modes (load along x, and load
along the transverse axis via a +90 deg layup rotation). Metrics are reported on
the load-direction plies for each mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

QUASI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(QUASI_DIR.parents[1]))
sys.path.insert(0, str(QUASI_DIR))

from layups import (  # noqa: E402
    LOAD_MODES,
    base_quasi_params,
    load_mode_params,
    set_group_gap_shifts,
    set_ply_stagger,
)
from springshear.metrics.objectives import evaluate_objective  # noqa: E402
from springshear.solve.run import solve  # noqa: E402

METRIC_KEYS = (
    "stress_cv",
    "stress_cv_abs",
    "mean_tow_stress",
    "tau_max",
    "E_eff",
    "physically_sane",
)


def run_preset(layup: str, mode: str, load_mode: str) -> dict:
    params = base_quasi_params(layup, bc_mode="fixed")
    if mode == "aligned":
        set_ply_stagger(params, 0.0, 0.0)
        set_group_gap_shifts(params, {"+0": 0.0, "+90": 0.0, "+45": 0.0, "-45": 0.0})
    elif mode == "half_gap":
        set_group_gap_shifts(params, {"+0": 0.5, "+90": 0.5, "+45": 0.5, "-45": 0.5})
    else:
        raise ValueError(mode)

    solve_params = load_mode_params(params, load_mode)
    x, u, elems, _ = solve(solve_params)
    metrics = evaluate_objective(solve_params, x, u, elems)
    metrics["layup"] = layup
    metrics["mode"] = mode
    metrics["load_mode"] = load_mode
    return metrics


def print_table(layup: str, load_mode: str, results: list[dict]) -> None:
    print(f"\n{layup}   |   load mode: {load_mode}")
    print(f"{'gap mode':<12}", end="")
    for key in METRIC_KEYS:
        print(f"{key:>18}", end="")
    print()
    print("-" * (12 + 18 * len(METRIC_KEYS)))
    for m in results:
        print(f"{m['mode']:<12}", end="")
        for key in METRIC_KEYS:
            val = m[key]
            if isinstance(val, bool):
                print(f"{str(val):>18}", end="")
            else:
                print(f"{val:18.4e}", end="")
        print()


def main():
    layups = ("[0/90]s", "[0/45/90]s", "[0/45/90]")
    for layup in layups:
        for load_mode in LOAD_MODES:
            aligned = run_preset(layup, "aligned", load_mode)
            half = run_preset(layup, "half_gap", load_mode)
            print_table(layup, load_mode, [aligned, half])
            if aligned["physically_sane"] and half["physically_sane"]:
                if aligned["stress_cv"] > 0:
                    pct = (aligned["stress_cv"] - half["stress_cv"]) / aligned["stress_cv"] * 100
                    print(f"  half_gap CV change vs aligned: {pct:+.1f}%")
            else:
                print("  WARNING: at least one case is not physically sane")


if __name__ == "__main__":
    main()
