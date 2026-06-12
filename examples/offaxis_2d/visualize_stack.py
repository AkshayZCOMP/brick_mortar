"""Visualize [0]4 stack gap placement: through-thickness layout + tow stress.

Produces three figures:
  * stack_layout_0_4.png  - schematic x-z cross-section of the [0]4 stack showing
    where the resin gaps sit in each ply, aligned vs staggered through thickness.
  * viz2d_0_4_aligned.png / viz2d_0_4_staggered.png - per-ply in-plane tow stress
    fields (same style as the QI stress plots), so the [0]4 baseline and best
    placements can be seen directly.

Run from the repo root:
    $env:MPLBACKEND="Agg"; python examples/offaxis_2d/visualize_stack.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from springshear.spring2d.metrics2d import evaluate2d  # noqa: E402
from springshear.spring2d.params2d import Params2D  # noqa: E402
from springshear.spring2d.solve2d import solve2d  # noqa: E402

# Reuse the per-ply stress-field renderer.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_2d import visualize  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
N_PLIES = 4


def _draw_stack(ax, gap_shifts: list[float], n_periods: int, title: str) -> None:
    base = Params2D()
    period, L_fiber, gap = base.period, base.L_fiber, base.resin_gap
    t_ply, t_il = base.t_ply, base.interlaminar_thickness
    Lx = n_periods * period
    ply_pitch = t_ply + t_il

    for i in range(N_PLIES):
        z0 = i * ply_pitch
        # interlaminar resin layer below each ply (except the first)
        if i > 0:
            ax.add_patch(Rectangle((0, z0 - t_il), Lx, t_il, facecolor="0.8", edgecolor="none"))
        # fibre band
        ax.add_patch(Rectangle((0, z0), Lx, t_ply, facecolor="#2c6fbb", edgecolor="none"))
        # severed gaps (resin) as red blocks
        shift = gap_shifts[i] * period
        m = -1
        while True:
            gx = m * period + L_fiber + shift
            if gx > Lx:
                break
            if gx + gap > 0:
                x0 = max(0.0, gx)
                x1 = min(Lx, gx + gap)
                if x1 > x0:
                    ax.add_patch(Rectangle((x0, z0), x1 - x0, t_ply, facecolor="#d62728", edgecolor="none"))
            m += 1
        ax.text(-0.02 * Lx, z0 + t_ply / 2, f"ply {i}", ha="right", va="center", fontsize=8)

    ax.set_xlim(0, Lx)
    ax.set_ylim(-t_il, N_PLIES * ply_pitch)
    ax.set_xlabel("x  (along fibre, m)")
    ax.set_ylabel("through-thickness z (m)")
    ax.set_title(title, fontsize=10)


def stack_layout() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), layout="constrained")
    _draw_stack(axes[0], [0.0, 0.0, 0.0, 0.0], n_periods=4,
                title="[0]4 gaps ALIGNED through thickness (s=0): gaps stack -> no bridging")
    _draw_stack(axes[1], [0.0, 0.25, 0.5, 0.75], n_periods=4,
                title="[0]4 gaps STAGGERED through thickness (s=1/4): neighbours bridge each gap")
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#2c6fbb"),
        Rectangle((0, 0), 1, 1, facecolor="#d62728"),
        Rectangle((0, 0), 1, 1, facecolor="0.8"),
    ]
    fig.legend(handles, ["intact tow (fibre)", "severed resin gap", "interlaminar resin"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    out = OUT_DIR / "stack_layout_0_4.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def stress_field(gap_shifts: list[float], tag: str) -> None:
    base = Params2D()
    params = Params2D(
        Lx=6 * base.period,
        Ly=6 * base.pitch,
        ply_angles=[0.0] * N_PLIES,
        ply_gap_shifts=gap_shifts,
        load_dir="x",
    )
    mesh, u = solve2d(params)
    metrics = evaluate2d(params, mesh, u)
    out = OUT_DIR / f"viz2d_0_4_{tag}.png"
    title = (f"[0]4  |  x-tension  |  {tag} gaps  |  "
             f"load-dir CV={metrics['stress_cv']:.3f}, E_eff context in compare_stiffness")
    visualize(params, mesh, u, title, out)


def main() -> None:
    stack_layout()
    stress_field([0.0, 0.0, 0.0, 0.0], "aligned")
    stress_field([0.0, 0.25, 0.5, 0.75], "staggered")


if __name__ == "__main__":
    main()
