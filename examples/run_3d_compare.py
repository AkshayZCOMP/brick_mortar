"""Compare aligned vs half-ply stagger for 2-ply fixed-end tension."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from springshear.metrics.objectives import evaluate_objective
from springshear.params import Params
from springshear.post.plots import (
    plot_fixed_tension_compare,
    plot_fixed_tension_heatmap_compare,
    plot_stack_stresses,
    plot_stress_heatmap,
)
from springshear.solve.run import solve

METRIC_KEYS = (
    "stress_cv",
    "stress_cv_abs",
    "mean_tow_stress",
    "tau_max",
    "E_eff",
    "physically_sane",
)


def run_preset(mode: str) -> dict:
    params = Params(
        n_plies=2,
        n_fibers=2,
        n_rows=3,
        dx=0.2e-3,
        bc_mode="fixed",
        periodic_x=False,
        periodic_y=False,
        periodic_z=False,
    )
    params.apply_stagger_preset(mode)
    x, u, elems, _ = solve(params)
    metrics = evaluate_objective(params, x, u, elems)
    metrics["mode"] = mode
    return {"params": params, "x": x, "u": u, "elems": elems, "metrics": metrics}


def print_table(results: list[dict]) -> None:
    print(f"\n{'mode':<12}", end="")
    for key in METRIC_KEYS:
        print(f"{key:>18}", end="")
    print()
    print("-" * (12 + 18 * len(METRIC_KEYS)))

    for r in results:
        m = r["metrics"]
        print(f"{m['mode']:<12}", end="")
        for key in METRIC_KEYS:
            val = m[key]
            if isinstance(val, bool):
                print(f"{str(val):>18}", end="")
            elif isinstance(val, float):
                print(f"{val:18.4e}", end="")
            else:
                print(f"{val!s:>18}", end="")
        print()


def main():
    out_dir = Path(__file__).parent
    aligned = run_preset("aligned")
    half_ply = run_preset("half_ply")
    results = [aligned, half_ply]

    print("3D stagger comparison (2-ply, fixed-end tension)")
    print_table(results)

    cv_aligned = aligned["metrics"]["stress_cv"]
    cv_half = half_ply["metrics"]["stress_cv"]
    improvement = (cv_aligned - cv_half) / cv_aligned * 100 if cv_aligned > 0 else 0.0
    print(f"\nStress CV improvement (half_ply vs aligned): {improvement:.1f}%")

    if not aligned["metrics"]["physically_sane"]:
        raise SystemExit("FAIL: aligned case is not physically sane")
    if not half_ply["metrics"]["physically_sane"]:
        raise SystemExit("FAIL: half_ply case is not physically sane")
    if cv_half >= cv_aligned:
        raise SystemExit("FAIL: half_ply stress_cv should be lower than aligned")

    print("PASS: half_ply is physically sane and reduces stress CV vs aligned")

    compare_png = out_dir / "compare_fixed_tension.png"
    plot_fixed_tension_compare(aligned, half_ply, out_path=str(compare_png), show=True)
    print(f"Wrote {compare_png}")

    heatmap_png = out_dir / "heatmap_fixed_tension.png"
    plot_fixed_tension_heatmap_compare(
        aligned, half_ply, out_path=str(heatmap_png), show=True
    )
    print(f"Wrote {heatmap_png}")

    for mode, case in [("aligned", aligned), ("half_ply", half_ply)]:
        stack_png = out_dir / f"stack_{mode}_fixed.png"
        fig = plot_stack_stresses(
            case["params"], case["x"], case["u"], case["elems"], show=False
        )
        fig.savefig(stack_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {stack_png}")

        single_hm = out_dir / f"heatmap_{mode}_fixed.png"
        hm_fig, _, _ = plot_stress_heatmap(
            case["params"],
            case["x"],
            case["u"],
            case["elems"],
            title=f"{mode} — axial stress (FEA view)",
            out_path=str(single_hm),
            show=False,
        )
        plt.close(hm_fig)
        print(f"Wrote {single_hm}")


if __name__ == "__main__":
    main()
