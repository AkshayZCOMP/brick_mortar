"""2D gap-placement sweep: longitudinal gap shift x transverse per-layer shift.

For a quasi-isotropic layup at the bulk-converged RVE size, this sweeps the two
independent stacking degrees of freedom of the 2D off-axis model:

  * s_long  - longitudinal shift of each layer's gap pattern ALONG the tow axis
              (ply i gets i * s_long * period); staggers brick gaps through the
              thickness in the fiber direction.
  * s_trans - transverse shift of each layer's tow-strip lattice PERPENDICULAR to
              the tows (ply i gets i * s_trans * pitch); offsets the rows of one
              layer relative to another so tow boundaries do not stack.

Each configuration is solved in both x- and y-tension; the objective is the
worst-case load-direction tow-stress CV across the two modes (lower = more
uniform = better). Outputs a heatmap PNG and a CSV, and reports the best cell.

Run from the repo root:
    $env:MPLBACKEND="Agg"; python examples/offaxis_2d/sweep_placement.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from springshear.spring2d.metrics2d import evaluate2d  # noqa: E402
from springshear.spring2d.params2d import Params2D  # noqa: E402
from springshear.spring2d.solve2d import solve2d  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

LAYUP_NAME = "[0/45/90]s"
LAYUP_ANGLES = [0.0, 45.0, -45.0, 90.0]
NX_TOWS, NY_TOWS = 10, 16
SHIFTS = [0.0, 0.25, 0.5, 0.75]
LOAD_MODES = ("x", "y")


def _params(s_long: float, s_trans: float, load_dir: str) -> Params2D:
    base = Params2D(ply_angles=LAYUP_ANGLES)
    n = len(LAYUP_ANGLES)
    return Params2D(
        Lx=NX_TOWS * base.period,
        Ly=NY_TOWS * base.pitch,
        ply_angles=LAYUP_ANGLES,
        ply_gap_shifts=[(i * s_long) % 1.0 for i in range(n)],
        ply_transverse_shifts=[(i * s_trans) % 1.0 for i in range(n)],
        load_dir=load_dir,
    )


def worst_case_cv(s_long: float, s_trans: float) -> dict:
    cvs = {}
    for mode in LOAD_MODES:
        params = _params(s_long, s_trans, mode)
        mesh, u = solve2d(params)
        cvs[mode] = evaluate2d(params, mesh, u)["stress_cv"]
    return {
        "s_long": s_long,
        "s_trans": s_trans,
        "cv_x": cvs["x"],
        "cv_y": cvs["y"],
        "cv_worst": max(cvs.values()),
    }


def main() -> None:
    print(f"2D placement sweep for {LAYUP_NAME} at {NX_TOWS}x{NY_TOWS} tows "
          f"(objective: worst-case load-dir CV over x/y tension)\n")
    grid = np.full((len(SHIFTS), len(SHIFTS)), np.nan)
    rows: list[dict] = []
    best = None
    for i, s_long in enumerate(SHIFTS):
        for j, s_trans in enumerate(SHIFTS):
            r = worst_case_cv(s_long, s_trans)
            rows.append(r)
            grid[i, j] = r["cv_worst"]
            print(f"  s_long={s_long:.2f} s_trans={s_trans:.2f}  "
                  f"cv_x={r['cv_x']:.4f} cv_y={r['cv_y']:.4f} cv_worst={r['cv_worst']:.4f}")
            if best is None or r["cv_worst"] < best["cv_worst"]:
                best = r

    csv_path = OUT_DIR / "sweep_placement.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["s_long", "s_trans", "cv_x", "cv_y", "cv_worst"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    im = ax.imshow(grid, origin="lower", cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(SHIFTS)), [f"{s:.2f}" for s in SHIFTS])
    ax.set_yticks(range(len(SHIFTS)), [f"{s:.2f}" for s in SHIFTS])
    ax.set_xlabel("transverse per-layer shift  s_trans  (fraction of pitch)")
    ax.set_ylabel("longitudinal gap shift  s_long  (fraction of period)")
    for i in range(len(SHIFTS)):
        for j in range(len(SHIFTS)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                    color="white", fontsize=9)
    bi = SHIFTS.index(best["s_long"])
    bj = SHIFTS.index(best["s_trans"])
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.5))
    fig.colorbar(im, ax=ax, label="worst-case load-dir tow-stress CV")
    ax.set_title(f"{LAYUP_NAME}: gap-placement sweep ({NX_TOWS}x{NY_TOWS} tows)\n"
                 f"red = best (s_long={best['s_long']:.2f}, s_trans={best['s_trans']:.2f}, "
                 f"CV={best['cv_worst']:.3f})")
    out_path = OUT_DIR / "sweep_placement.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    print("\n" + "=" * 60)
    print(f"BEST PLACEMENT: s_long={best['s_long']:.2f}, s_trans={best['s_trans']:.2f}")
    print(f"  worst-case load-dir CV = {best['cv_worst']:.4f} "
          f"(cv_x={best['cv_x']:.4f}, cv_y={best['cv_y']:.4f})")
    aligned = next(r for r in rows if r["s_long"] == 0.0 and r["s_trans"] == 0.0)
    impr = (aligned["cv_worst"] - best["cv_worst"]) / aligned["cv_worst"] * 100
    print(f"  vs fully aligned (0,0) CV={aligned['cv_worst']:.4f}: {impr:+.1f}% CV reduction")
    print("=" * 60)


if __name__ == "__main__":
    main()
