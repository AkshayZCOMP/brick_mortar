"""Parametric sweep of 3D ply stagger offsets with periodic BCs."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from springshear.metrics.objectives import evaluate_objective
from springshear.params import Params
from springshear.solve.run import solve


def run_case(offset_x_frac: float, offset_y_frac: float, n_plies: int = 2) -> dict:
    params = Params(
        n_plies=n_plies,
        n_fibers=2,
        n_rows=3,
        dx=0.2e-3,
        bc_mode="periodic",
        periodic_x=True,
        periodic_y=False,
        periodic_z=False,
    )
    params.ply_offset_x = offset_x_frac * params.L_fiber
    params.ply_offset_y = offset_y_frac * params.row_pitch
    x, u, elems, _ = solve(params)
    metrics = evaluate_objective(params, x, u, elems)
    metrics["offset_x_frac"] = offset_x_frac
    metrics["offset_y_frac"] = offset_y_frac
    return metrics


def main():
    n_steps = 7
    xs = np.linspace(0.0, 1.0, n_steps)
    ys = np.linspace(0.0, 1.0, n_steps)
    results = []

    print(f"Sweeping {n_steps}x{n_steps} grid ...")
    for ox in xs:
        for oy in ys:
            m = run_case(float(ox), float(oy))
            results.append(m)
            print(
                f"  ox={ox:.2f} oy={oy:.2f}  CV={m['stress_cv']:.4f}  "
                f"CV_abs={m['stress_cv_abs']:.4f}  sane={m['physically_sane']}  "
                f"E_eff={m['E_eff']:.3e}  tau_max={m['tau_max']:.3e}"
            )

    out_dir = Path(__file__).parent
    out_csv = out_dir / "sweep_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {out_csv}")

    cv_grid = np.array(
        [[r["stress_cv"] for r in results if abs(r["offset_x_frac"] - ox) < 1e-9] for ox in xs]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        cv_grid,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="viridis_r",
    )
    plt.colorbar(im, ax=ax, label="Stress CV (interior tow axial)")
    ax.scatter([0.5], [0.5], c="red", s=80, marker="x", label="half-offset hypothesis")
    best = min(results, key=lambda r: r["stress_cv"])
    ax.scatter(
        [best["offset_x_frac"]],
        [best["offset_y_frac"]],
        c="white",
        edgecolors="black",
        s=80,
        marker="*",
        label=f"best ({best['offset_x_frac']:.2f}, {best['offset_y_frac']:.2f})",
    )
    ax.set(xlabel="ply_offset_x / L_fiber", ylabel="ply_offset_y / row_pitch", title="3D stagger sweep")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sweep_heatmap.png", dpi=150)
    print(f"Wrote {out_dir / 'sweep_heatmap.png'}")
    print(f"Best: ox={best['offset_x_frac']:.2f}, oy={best['offset_y_frac']:.2f}, CV={best['stress_cv']:.4f}")
    hyp = min(
        results,
        key=lambda r: abs(r["offset_x_frac"] - 0.5) + abs(r["offset_y_frac"] - 0.5),
    )
    aligned = min(
        results,
        key=lambda r: abs(r["offset_x_frac"]) + abs(r["offset_y_frac"]),
    )
    print(f"Near hypothesis (0.5,0.5): ox={hyp['offset_x_frac']:.2f} oy={hyp['offset_y_frac']:.2f} CV={hyp['stress_cv']:.4f}")
    print(
        f"Aligned (0,0): CV={aligned['stress_cv']:.4f}  sane={aligned['physically_sane']}  "
        f"tau_max={aligned['tau_max']:.3e}"
    )
    if aligned["stress_cv"] > 0:
        pct = (aligned["stress_cv"] - hyp["stress_cv"]) / aligned["stress_cv"] * 100
        print(f"Half-offset CV reduction vs aligned: {pct:.1f}%")
    plt.close(fig)


if __name__ == "__main__":
    main()
