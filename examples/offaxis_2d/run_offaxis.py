"""Drive the 2D off-axis spring solver for QI layups in x and y tension.

Reports load-direction tow-stress CV and peak matrix shear for each layup under
both in-plane tension modes, comparing aligned vs half-period gap placement.

Run from the repo root:

    python examples/offaxis_2d/run_offaxis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from springshear.spring2d.metrics2d import evaluate2d  # noqa: E402
from springshear.spring2d.params2d import Params2D  # noqa: E402
from springshear.spring2d.solve2d import solve2d  # noqa: E402

LAYUPS = {
    "[0/90]s": [0.0, 90.0, 90.0, 0.0],
    "[0/45/90]s": [0.0, 45.0, -45.0, 90.0],
    "[0/45/90]": [0.0, 45.0, 90.0],
}
METRIC_KEYS = ("stress_cv", "mean_tow_stress", "max_tow_stress", "tau_matrix_max", "n_load_tows")

# Bulk-converged RVE size (see convergence.py): fixed-grip BCs need a long
# gauge in the load direction; boundary layers otherwise inflate the response.
NX_TOWS = 10  # brick-periods along the load direction
NY_TOWS = 16  # tow strips across the width


def run_case(angles: list[float], gap_shift: float, load_dir: str) -> dict:
    base = Params2D(ply_angles=angles)
    params = Params2D(
        Lx=NX_TOWS * base.period,
        Ly=NY_TOWS * base.pitch,
        ply_angles=angles,
        ply_gap_shifts=[gap_shift] * len(angles),
        load_dir=load_dir,
    )
    mesh, u = solve2d(params)
    metrics = evaluate2d(params, mesh, u)
    metrics["gap_shift"] = gap_shift
    return metrics


def print_table(layup: str, load_dir: str, rows: list[dict]) -> None:
    print(f"\n{layup}   |   {load_dir}-tension")
    print(f"{'gap':<10}", end="")
    for key in METRIC_KEYS:
        print(f"{key:>16}", end="")
    print()
    print("-" * (10 + 16 * len(METRIC_KEYS)))
    for r in rows:
        label = "aligned" if r["gap_shift"] == 0.0 else f"shift {r['gap_shift']:.2f}"
        print(f"{label:<10}", end="")
        for key in METRIC_KEYS:
            val = r[key]
            print(f"{val:16.4e}" if isinstance(val, float) else f"{val:>16}", end="")
        print()


def main() -> None:
    for layup, angles in LAYUPS.items():
        for load_dir in ("x", "y"):
            aligned = run_case(angles, 0.0, load_dir)
            half = run_case(angles, 0.5, load_dir)
            print_table(layup, load_dir, [aligned, half])
            if aligned["stress_cv"] > 0 and aligned["physically_sane"] and half["physically_sane"]:
                pct = (aligned["stress_cv"] - half["stress_cv"]) / aligned["stress_cv"] * 100
                print(f"  half-shift CV change vs aligned: {pct:+.1f}%")


if __name__ == "__main__":
    main()
