"""Visualize fiber orientation, gap placement, and stress for non-UD layups.

For each ply the figure shows two panels:

  * Left  - geometry map: tow bands shaded by fiber integrity (intact fiber vs
            resin gap, using the *same* gap logic the assembler uses), with an
            arrow marking the fiber orientation theta.
  * Right - axial stress field recovered from the solved displacement, on the
            same band layout.

This makes the per-orientation gap-staggering question visible: 0 deg gaps form
vertical stripes, 90 deg breaks fall on row boundaries, and +/-45 deg gaps form
diagonal stripes because the brick pattern is projected onto the fiber axis.

Run (from the repo root):

    python examples/quasi_isotropic/visualize_layup.py

Headless / server use:

    # PowerShell
    $env:MPLBACKEND="Agg"; python examples/quasi_isotropic/visualize_layup.py
    # bash
    MPLBACKEND=Agg python examples/quasi_isotropic/visualize_layup.py

PNGs are written next to this script (one per layup x gap-config).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrow

QUASI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(QUASI_DIR.parents[1]))
sys.path.insert(0, str(QUASI_DIR))

from layups import base_quasi_params, load_mode_params, set_group_gap_shifts  # noqa: E402
from springshear.geometry.staggering import (  # noqa: E402
    fiber_strength,
    ply_y_shift,
)
from springshear.params import Params  # noqa: E402
from springshear.solve.run import solve  # noqa: E402

OUT_DIR = QUASI_DIR


def fiber_integrity_field(params: Params, x: np.ndarray, ply: int) -> np.ndarray:
    """Return fiber integrity (1 = intact tow, ~0 = resin gap) per (row, segment)."""
    x_mid = 0.5 * (x[:-1] + x[1:])
    field = np.ones((params.n_rows, len(x_mid)))
    for row in range(params.n_rows):
        for j, xm in enumerate(x_mid):
            field[row, j] = fiber_strength(params, ply, row, float(xm))
    return field


def ply_stress_field(params: Params, x: np.ndarray, u: np.ndarray, elems: list, ply: int) -> np.ndarray:
    """Tow/bridge axial stress per (row, segment) in Pa.

    Robust to both axial plies (x-segment tows, indices 0..n_seg-1) and
    transverse plies (per-node y-link tows, indices 0..n_node-1, clamped).
    """
    n_seg = len(x) - 1
    n_rows = params.n_rows
    field = np.full((n_rows, n_seg), np.nan)
    A_m = params.A_m
    for e in elems:
        if e.get("ply", 0) != ply:
            continue
        if e["etype"] not in ("tow_axial", "tow_bridge"):
            continue
        force = e["k"] * (u[e["b"]] - u[e["a"]])
        area = e["A"] if e["etype"] == "tow_axial" else A_m
        sigma = force / area
        seg = min(int(e["i"]), n_seg - 1)
        row = int(e["row"])
        # Transverse (y-link) tows span two adjacent bands; fill both so the
        # 90 deg ply renders fully instead of leaving the last band blank.
        rows = [row, row + 1] if e.get("direction") == "y" else [row]
        for r in rows:
            r_mod = r % n_rows if params.periodic_y else r
            if 0 <= r_mod < n_rows:
                field[r_mod, seg] = sigma
    return field


def _band_y(params: Params, ply: int, row: int) -> tuple[float, float]:
    y0 = ply_y_shift(params, ply) + row * params.row_pitch
    return y0, y0 + params.row_width


def _fiber_arrow(ax, angle_deg: float, x_center: float, y_center: float, length: float) -> None:
    theta = np.deg2rad(angle_deg)
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    ax.add_patch(
        FancyArrow(
            x_center - 0.5 * dx,
            y_center - 0.5 * dy,
            dx,
            dy,
            width=length * 0.012,
            head_width=length * 0.06,
            head_length=length * 0.06,
            length_includes_head=True,
            color="black",
            zorder=6,
        )
    )


def visualize(
    params: Params,
    x: np.ndarray,
    u: np.ndarray,
    elems: list,
    title: str,
    out_path: Path,
) -> None:
    n_plies = params.n_plies
    x_edges = x

    stress_fields = [ply_stress_field(params, x, u, elems, p) for p in range(n_plies)]
    finite = np.concatenate([f[np.isfinite(f)] for f in stress_fields if np.isfinite(f).any()])
    vmin, vmax = float(finite.min()) * 1e-9, float(finite.max()) * 1e-9
    snorm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        n_plies,
        2,
        figsize=(13, 1.9 * n_plies + 1.5),
        squeeze=False,
        layout="constrained",
    )

    x_span = float(x_edges[-1] - x_edges[0])
    pcm_stress = None
    for ply in range(n_plies):
        ax_geo, ax_sig = axes[ply]
        integ = fiber_integrity_field(params, x, ply)
        sig = stress_fields[ply] * 1e-9
        angle = params.ply_angle(ply)

        y_lo, y_hi = _band_y(params, ply, 0)
        for row in range(params.n_rows):
            y0, y1 = _band_y(params, ply, row)
            ax_geo.pcolormesh(
                x_edges,
                [y0, y1],
                integ[row].reshape(1, -1),
                cmap="Greys",
                vmin=0.0,
                vmax=1.0,
                shading="flat",
                zorder=1,
            )
            pcm_stress = ax_sig.pcolormesh(
                x_edges,
                [y0, y1],
                sig[row].reshape(1, -1),
                cmap="jet",
                norm=snorm,
                shading="flat",
                zorder=1,
            )

        y_top = _band_y(params, ply, params.n_rows - 1)[1]
        for ax in (ax_geo, ax_sig):
            ax.set_xlim(x_edges[0], x_edges[-1])
            ax.set_ylim(y_lo, y_top)
            ax.set_ylabel("y (m)")
        _fiber_arrow(ax_geo, angle, 0.5 * x_span, 0.5 * (y_lo + y_top), 0.18 * x_span)
        ax_geo.set_title(f"Ply {ply}: theta = {angle:+.0f} deg  (white = resin gap)", fontsize=9)
        ax_sig.set_title(f"Ply {ply}: axial stress (GPa)", fontsize=9)

    for ax in axes[-1]:
        ax.set_xlabel("x (m) - load direction")

    if pcm_stress is not None:
        fig.colorbar(pcm_stress, ax=axes[:, 1].tolist(), label="axial stress (GPa)", fraction=0.05)

    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def visualize_layup(layup: str, gap_shift: float, load_mode: str = "0deg") -> None:
    params = base_quasi_params(layup, bc_mode="fixed")
    shifts = {"+0": gap_shift, "+90": gap_shift, "+45": gap_shift, "-45": gap_shift}
    set_group_gap_shifts(params, shifts)
    solve_params = load_mode_params(params, load_mode)
    x, u, elems, _ = solve(solve_params)

    slug = layup.replace("[", "").replace("]", "").replace("/", "_").replace(" ", "")
    tag = "aligned" if gap_shift == 0.0 else f"shift{gap_shift:.2f}".replace(".", "p")
    out_path = OUT_DIR / f"viz_{slug}_{tag}_{load_mode}.png"
    title = (
        f"{layup}  |  {load_mode} tension, eps0={params.eps0:.3f}  |  "
        f"per-orientation gap shift = {gap_shift:.2f} period"
    )
    visualize(solve_params, x, u, elems, title, out_path)


def main() -> None:
    # Visualize both tension modes so the load-direction swap is visible.
    cases = [
        ("[0/90]s", 0.0, "0deg"),
        ("[0/45/90]s", 0.0, "0deg"),
        ("[0/45/90]s", 0.0, "90deg"),
    ]
    for layup, shift, load_mode in cases:
        print(f"Solving {layup} (gap shift {shift:.2f}, {load_mode}) ...")
        visualize_layup(layup, shift, load_mode)


if __name__ == "__main__":
    main()
