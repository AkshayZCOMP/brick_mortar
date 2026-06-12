"""Sweep per-orientation gap shifts for quasi-isotropic layups."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

QUASI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(QUASI_DIR.parents[1]))
sys.path.insert(0, str(QUASI_DIR))

from layups import (
    LOAD_MODES,
    angle_groups,
    base_quasi_params,
    load_mode_params,
    set_group_gap_shifts,
    set_ply_stagger,
)
from springshear.metrics.objectives import evaluate_objective
from springshear.solve.run import solve

OUT_DIR = QUASI_DIR

# Objective key the sweep minimizes. cv_worst = worst (max) stress CV across the
# 0 deg and 90 deg tension modes, so the chosen gap placement is robust to both.
OBJECTIVE = "cv_worst"


def run_case(
    layup: str,
    shift_0: float,
    shift_90: float,
    shift_45: float = 0.0,
    offset_x_frac: float = 0.0,
    offset_y_frac: float = 0.0,
) -> dict:
    params = base_quasi_params(layup)
    if offset_x_frac or offset_y_frac:
        set_ply_stagger(params, offset_x_frac, offset_y_frac)
    shifts = {"+0": shift_0, "+90": shift_90}
    if any(abs(a) == 45 for a in params.ply_angles):
        shifts["+45"] = shift_45
        shifts["-45"] = shift_45
    set_group_gap_shifts(params, shifts)

    result: dict = {
        "layup": layup,
        "shift_0": shift_0,
        "shift_90": shift_90,
        "shift_45": shift_45,
        "offset_x_frac": offset_x_frac,
        "offset_y_frac": offset_y_frac,
        "ply_angles": ",".join(str(int(a)) for a in params.ply_angles),
    }

    cvs: list[float] = []
    sane = True
    for load_mode in LOAD_MODES:
        solve_params = load_mode_params(params, load_mode)
        x, u, elems, _ = solve(solve_params)
        m = evaluate_objective(solve_params, x, u, elems)
        result[f"cv_{load_mode}"] = m["stress_cv"]
        result[f"mean_stress_{load_mode}"] = m["mean_tow_stress"]
        result[f"tau_max_{load_mode}"] = m["tau_max"]
        result[f"sane_{load_mode}"] = m["physically_sane"]
        cvs.append(m["stress_cv"])
        sane = sane and m["physically_sane"]

    result["cv_worst"] = max(cvs)
    result["cv_mean"] = float(np.mean(cvs))
    result["physically_sane"] = sane
    result["stress_cv"] = result["cv_worst"]
    return result


def sweep_layup(layup: str, n_steps: int = 5) -> list[dict]:
    shifts = np.linspace(0.0, 1.0, n_steps)
    results: list[dict] = []
    has_45 = layup in ("[0/45/90]s", "quasi_isotropic", "[0/45/90]")

    print(f"\n=== {layup} gap-placement sweep ({n_steps} steps/orientation) ===")
    print("  (CV reported per tension mode; objective = worst-case across modes)")
    for s0 in shifts:
        for s90 in shifts:
            s45 = 0.5 if has_45 else 0.0
            if has_45:
                for s45 in shifts:
                    m = run_case(layup, float(s0), float(s90), float(s45))
                    results.append(m)
                    print(
                        f"  s0={s0:.2f} s90={s90:.2f} s45={s45:.2f}  "
                        f"CV[0]={m['cv_0deg']:.4f} CV[90]={m['cv_90deg']:.4f} "
                        f"worst={m['cv_worst']:.4f}  sane={m['physically_sane']}"
                    )
            else:
                m = run_case(layup, float(s0), float(s90))
                results.append(m)
                print(
                    f"  s0={s0:.2f} s90={s90:.2f}  "
                    f"CV[0]={m['cv_0deg']:.4f} CV[90]={m['cv_90deg']:.4f} "
                    f"worst={m['cv_worst']:.4f}  sane={m['physically_sane']}"
                )
    return results


def save_results(layup: str, results: list[dict]) -> Path:
    slug = layup.replace("[", "").replace("]", "").replace("/", "_").replace(" ", "")
    out_csv = OUT_DIR / f"sweep_{slug}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {out_csv}")
    return out_csv


def plot_0_90_slice(layup: str, results: list[dict], fixed_45: float | None = None) -> None:
    if layup not in ("[0/90]s",):
        return
    n = len({r["shift_0"] for r in results})
    cv_grid = np.full((n, n), np.nan)
    xs = sorted({r["shift_0"] for r in results})
    ys = sorted({r["shift_90"] for r in results})
    for r in results:
        ix = xs.index(r["shift_0"])
        iy = ys.index(r["shift_90"])
        cv_grid[iy, ix] = r["stress_cv"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        cv_grid,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        aspect="auto",
        cmap="viridis_r",
    )
    plt.colorbar(im, ax=ax, label="Worst-case stress CV (0 deg & 90 deg modes)")
    best = min(results, key=lambda r: r["stress_cv"])
    ax.scatter(
        [best["shift_0"]],
        [best["shift_90"]],
        c="white",
        edgecolors="black",
        s=80,
        marker="*",
        label=f"best ({best['shift_0']:.2f}, {best['shift_90']:.2f})",
    )
    ax.scatter([0.5], [0.5], c="red", s=60, marker="x", label="half-period hypothesis")
    ax.set(
        xlabel="0° ply gap shift / period",
        ylabel="90° ply gap shift / period",
        title=f"{layup} — gap placement sweep",
    )
    ax.legend()
    plt.tight_layout()
    out_png = OUT_DIR / "sweep_0_90s_heatmap.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")


def main():
    layups = ("[0/90]s", "[0/45/90]s")
    all_best: list[dict] = []

    for layup in layups:
        groups = angle_groups(base_quasi_params(layup))
        print(f"{layup}: orientation groups = {groups}")
        n_steps = 5 if layup == "[0/90]s" else 4
        results = sweep_layup(layup, n_steps=n_steps)
        save_results(layup, results)
        best = min(results, key=lambda r: r["stress_cv"])
        all_best.append(best)
        if layup == "[0/90]s":
            plot_0_90_slice(layup, results)

    print("\nBest gap placements found (minimizing worst-case CV across tension modes):")
    for b in all_best:
        print(
            f"  {b['layup']}: s0={b['shift_0']:.2f} s90={b['shift_90']:.2f} "
            f"s45={b['shift_45']:.2f}  "
            f"CV[0]={b['cv_0deg']:.4f} CV[90]={b['cv_90deg']:.4f} worst={b['cv_worst']:.4f}"
        )


if __name__ == "__main__":
    main()
