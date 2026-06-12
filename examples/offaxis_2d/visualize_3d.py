"""3D / through-ply schematics of the final optimized gap placements.

Draws each ply of a laminate as its true tow-strip pattern (continuous strips at
the ply angle, severed by resin gaps) on its own through-thickness plane, stacked
(exploded) so a viewer can see, ply by ply, exactly where the gaps sit and how the
optimized placement keeps them from lining up through the thickness.

Two staggering mechanisms are visible:
  * within a ply, adjacent tow rows are brick-staggered (strip_stagger_frac),
    so a gap in one row is bridged by intact tow in the neighbouring row;
  * between plies, the gap pattern is shifted longitudinally (ply_gap_shifts) and
    the row lattice is shifted transversely (ply_transverse_shifts), so gaps do
    not stack through the thickness.

Outputs:
  stack3d_0_4.png         - [0]4 baseline (aligned) vs optimized (staggered), 3D.
  stack3d_qi.png          - [0/45/90]s optimized, 3D.
  through_ply_0_4.png     - top-down per-ply maps (aligned vs optimized columns).

Run from the repo root:
    $env:MPLBACKEND="Agg"; python examples/offaxis_2d/visualize_3d.py
"""

from __future__ import annotations

import sys
from math import ceil, cos, floor, radians, sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from springshear.spring2d.params2d import Params2D  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

PLY_COLORS = {0.0: "#2c6fbb", 45.0: "#2ca02c", -45.0: "#17becf", 90.0: "#9467bd", -90.0: "#9467bd"}
GAP_COLOR = "#d62728"


def _clip_to_rect(poly: list[tuple[float, float]], Lx: float, Ly: float) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip of a convex polygon to [0,Lx] x [0,Ly]."""
    def clip_edge(pts, inside, intersect):
        out = []
        n = len(pts)
        for i in range(n):
            cur, prv = pts[i], pts[i - 1]
            ci, pi = inside(cur), inside(prv)
            if ci:
                if not pi:
                    out.append(intersect(prv, cur))
                out.append(cur)
            elif pi:
                out.append(intersect(prv, cur))
        return out

    def lerp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    pts = poly
    # x >= 0
    pts = clip_edge(pts, lambda p: p[0] >= 0, lambda a, b: lerp(a, b, (0 - a[0]) / (b[0] - a[0])))
    if not pts:
        return []
    pts = clip_edge(pts, lambda p: p[0] <= Lx, lambda a, b: lerp(a, b, (Lx - a[0]) / (b[0] - a[0])))
    if not pts:
        return []
    pts = clip_edge(pts, lambda p: p[1] >= 0, lambda a, b: lerp(a, b, (0 - a[1]) / (b[1] - a[1])))
    if not pts:
        return []
    pts = clip_edge(pts, lambda p: p[1] <= Ly, lambda a, b: lerp(a, b, (Ly - a[1]) / (b[1] - a[1])))
    return pts


def ply_polygons(params: Params2D, ply: int):
    """Return (fiber_polys, gap_polys) in x-y for one ply, clipped to the RVE."""
    theta = radians(params.ply_angle(ply))
    tx, ty = cos(theta), sin(theta)
    nx, ny = -sin(theta), cos(theta)
    Lx, Ly, pitch = params.Lx, params.Ly, params.pitch
    period, L_fiber, gap = params.period, params.L_fiber, params.resin_gap
    hw = params.tow_width / 2.0
    eta_shift = params.ply_transverse_shifts[ply] * pitch
    gap_shift0 = params.ply_gap_shifts[ply] * period

    corners = [(0.0, 0.0), (Lx, 0.0), (0.0, Ly), (Lx, Ly)]
    xis = [cx * tx + cy * ty for cx, cy in corners]
    etas = [cx * nx + cy * ny for cx, cy in corners]
    k_lo = floor((min(etas) - eta_shift) / pitch) - 1
    k_hi = ceil((max(etas) - eta_shift) / pitch) + 1
    xi_min, xi_max = min(xis), max(xis)

    fiber_polys, gap_polys = [], []
    for k in range(k_lo, k_hi + 1):
        eta = k * pitch + eta_shift
        shift = gap_shift0 + k * params.strip_stagger_frac * period
        # j range must follow the per-strip longitudinal shift, else high-k
        # strips (whose pattern is shifted far) get dropped and the ply tapers.
        j_lo = floor((xi_min - shift) / period) - 1
        j_hi = ceil((xi_max - shift) / period) + 1
        for j in range(j_lo, j_hi + 1):
            base = j * period + shift
            segments = [
                (base, base + L_fiber, fiber_polys),
                (base + L_fiber, base + period, gap_polys),
            ]
            for xi0, xi1, bucket in segments:
                quad = [
                    (xi0 * tx + (eta - hw) * nx, xi0 * ty + (eta - hw) * ny),
                    (xi1 * tx + (eta - hw) * nx, xi1 * ty + (eta - hw) * ny),
                    (xi1 * tx + (eta + hw) * nx, xi1 * ty + (eta + hw) * ny),
                    (xi0 * tx + (eta + hw) * nx, xi0 * ty + (eta + hw) * ny),
                ]
                clipped = _clip_to_rect(quad, Lx, Ly)
                if len(clipped) >= 3:
                    bucket.append(clipped)
    return fiber_polys, gap_polys


def _add_ply_3d(ax, params: Params2D, ply: int, z: float) -> None:
    fiber_polys, gap_polys = ply_polygons(params, ply)
    color = PLY_COLORS.get(params.ply_angle(ply), "#2c6fbb")
    fz = [[(x, y, z) for x, y in poly] for poly in fiber_polys]
    gz = [[(x, y, z) for x, y in poly] for poly in gap_polys]
    if fz:
        ax.add_collection3d(Poly3DCollection(fz, facecolors=color, edgecolors="none", alpha=0.92))
    if gz:
        ax.add_collection3d(Poly3DCollection(gz, facecolors=GAP_COLOR, edgecolors="none", alpha=1.0))


def _stack3d(ax, params: Params2D, title: str) -> None:
    # Exploded ply spacing in the same (metre) units as x/y so the box aspect
    # is physically sensible (otherwise the z axis dominates and squashes x-y).
    dz = 0.45 * params.Lx
    for ply in range(params.n_plies):
        _add_ply_3d(ax, params, ply, z=ply * dz)
        ax.text(0, params.Ly * 1.03, ply * dz,
                f"ply {ply} ({params.ply_angle(ply):+.0f} deg)", fontsize=8)
    ax.set_xlim(0, params.Lx)
    ax.set_ylim(0, params.Ly)
    ax.set_zlim(-0.5 * dz, (params.n_plies - 0.5) * dz)
    ax.set_box_aspect((params.Lx, params.Ly, params.n_plies * dz))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("through thickness (exploded)")
    ax.set_zticks([])
    ax.view_init(elev=20, azim=-60)
    ax.set_title(title, fontsize=11)


def _common(angles, shifts, trans, Lx, Ly):
    return Params2D(Lx=Lx, Ly=Ly, ply_angles=angles, ply_gap_shifts=shifts,
                    ply_transverse_shifts=trans)


def stack3d_0_4() -> None:
    base = Params2D()
    Lx, Ly = 3 * base.period, 8 * base.pitch
    aligned = _common([0.0] * 4, [0.0] * 4, [0.0] * 4, Lx, Ly)
    opt = _common([0.0] * 4, [0.0, 0.25, 0.5, 0.75], [0.0, 0.5, 0.0, 0.5], Lx, Ly)

    fig = plt.figure(figsize=(15, 6.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _stack3d(ax1, aligned, "[0]4 BASELINE: gaps aligned through thickness\n(gaps stack -> no bridging)")
    _stack3d(ax2, opt, "[0]4 OPTIMIZED: longitudinal + transverse stagger\n(gaps offset every ply -> neighbours bridge)")
    _legend(fig, opt.ply_angles)
    out = OUT_DIR / "stack3d_0_4.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def stack3d_qi() -> None:
    base = Params2D()
    Lx, Ly = 3 * base.period, 8 * base.pitch
    angles = [0.0, 45.0, -45.0, 90.0]
    opt = _common(angles, [0.0, 0.25, 0.5, 0.75], [0.0, 0.5, 0.0, 0.5], Lx, Ly)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _stack3d(ax, opt, "[0/45/90]s OPTIMIZED gap placement (one symmetric half shown)")
    _legend(fig, opt.ply_angles)
    out = OUT_DIR / "stack3d_qi.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _legend(fig, angles) -> None:
    handles = [MplPolygon([(0, 0)], closed=True, facecolor=PLY_COLORS.get(a, "#2c6fbb"),
                          label=f"tow {a:+.0f} deg") for a in dict.fromkeys(angles)]
    handles.append(MplPolygon([(0, 0)], closed=True, facecolor=GAP_COLOR, label="severed resin gap"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))


def through_ply_0_4() -> None:
    """Top-down per-ply maps: column A aligned, column B optimized (read down)."""
    base = Params2D()
    Lx, Ly = 3 * base.period, 8 * base.pitch
    aligned = _common([0.0] * 4, [0.0] * 4, [0.0] * 4, Lx, Ly)
    opt = _common([0.0] * 4, [0.0, 0.25, 0.5, 0.75], [0.0, 0.5, 0.0, 0.5], Lx, Ly)
    configs = [("aligned (baseline)", aligned), ("staggered (optimized)", opt)]

    fig, axes = plt.subplots(4, 2, figsize=(11, 10), layout="constrained")
    for col, (name, params) in enumerate(configs):
        for ply in range(4):
            ax = axes[ply, col]
            fiber_polys, gap_polys = ply_polygons(params, ply)
            for poly in fiber_polys:
                ax.add_patch(MplPolygon(poly, closed=True, facecolor=PLY_COLORS[0.0], edgecolor="none"))
            for poly in gap_polys:
                ax.add_patch(MplPolygon(poly, closed=True, facecolor=GAP_COLOR, edgecolor="none"))
            ax.set_xlim(0, Lx)
            ax.set_ylim(0, Ly)
            ax.set_aspect("equal")
            ax.set_title(f"{name}  -  ply {ply}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("[0]4 through-ply gap maps (read each column top->bottom). "
                 "Optimized column: gap bands shift every ply; aligned: they stack.", fontsize=11)
    out = OUT_DIR / "through_ply_0_4.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    stack3d_0_4()
    stack3d_qi()
    through_ply_0_4()


if __name__ == "__main__":
    main()
