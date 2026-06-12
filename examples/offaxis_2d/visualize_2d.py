"""Visualize the 2D off-axis spring network: tow stress fields and severed gaps.

For each ply a panel shows the tow strips (line segments) colored by axial
stress, with severed resin gaps drawn as light dashed segments. This makes the
2D load detour around severed off-axis tows directly visible.

Run from the repo root:

    python examples/offaxis_2d/visualize_2d.py
    # headless:
    $env:MPLBACKEND="Agg"; python examples/offaxis_2d/visualize_2d.py   # PowerShell
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from springshear.spring2d.metrics2d import evaluate2d  # noqa: E402
from springshear.spring2d.params2d import Params2D  # noqa: E402
from springshear.spring2d.solve2d import solve2d  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
LAYUPS = {
    "[0/90]s": [0.0, 90.0, 90.0, 0.0],
    "[0/45/90]s": [0.0, 45.0, -45.0, 90.0],
}


def _tow_segments_and_stress(params: Params2D, mesh, u: np.ndarray, ply: int):
    ux = u[0::2]
    uy = u[1::2]
    intact, intact_sig, severed = [], [], []
    for e in mesh.elements:
        if e.get("ply") != ply:
            continue
        pa, pb = mesh.nodes[e["a"]], mesh.nodes[e["b"]]
        if e["etype"] == "tow":
            strain = (
                (ux[e["b"]] - ux[e["a"]]) * e["dx"] + (uy[e["b"]] - uy[e["a"]]) * e["dy"]
            ) / params.dxi
            intact.append([pa, pb])
            intact_sig.append(params.E_tow * strain)
        elif e["etype"] == "tow_severed":
            severed.append([pa, pb])
    return intact, np.array(intact_sig), severed


def visualize(params: Params2D, mesh, u: np.ndarray, title: str, out_path: Path) -> None:
    n_plies = params.n_plies
    per = [_tow_segments_and_stress(params, mesh, u, p) for p in range(n_plies)]
    all_sig = np.concatenate([s for _, s, _ in per if s.size]) if any(s.size for _, s, _ in per) else np.array([0.0])
    norm = Normalize(vmin=float(all_sig.min()) * 1e-9, vmax=float(all_sig.max()) * 1e-9)

    ncol = min(n_plies, 2)
    nrow = int(np.ceil(n_plies / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 5.5 * nrow), squeeze=False, layout="constrained")
    axes = axes.ravel()

    lc = None
    for ply in range(n_plies):
        ax = axes[ply]
        intact, sig, severed = per[ply]
        if severed:
            ax.add_collection(LineCollection(severed, colors="0.7", linewidths=1.0, linestyles=":", zorder=1))
        if intact:
            lc = LineCollection(intact, cmap="jet", norm=norm, linewidths=2.2, zorder=2)
            lc.set_array(sig * 1e-9)
            ax.add_collection(lc)
        ax.set_xlim(0, params.Lx)
        ax.set_ylim(0, params.Ly)
        ax.set_aspect("equal")
        ax.set_title(f"Ply {ply}: theta={params.ply_angle(ply):+.0f} deg", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    for extra in range(n_plies, len(axes)):
        axes[extra].axis("off")

    if lc is not None:
        fig.colorbar(lc, ax=axes.tolist(), label="tow axial stress (GPa)", fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def visualize_layup(layup: str, angles: list[float], gap_shift: float, load_dir: str) -> None:
    params = Params2D(ply_angles=angles, ply_gap_shifts=[gap_shift] * len(angles), load_dir=load_dir)
    mesh, u = solve2d(params)
    metrics = evaluate2d(params, mesh, u)
    slug = layup.replace("[", "").replace("]", "").replace("/", "_")
    tag = "aligned" if gap_shift == 0.0 else f"shift{gap_shift:.2f}".replace(".", "p")
    out_path = OUT_DIR / f"viz2d_{slug}_{tag}_{load_dir}.png"
    title = (
        f"{layup}  |  {load_dir}-tension  |  gap shift {gap_shift:.2f}  |  "
        f"load-dir CV={metrics['stress_cv']:.3f}, tau_m,max={metrics['tau_matrix_max']/1e6:.1f} MPa"
    )
    visualize(params, mesh, u, title, out_path)


def main() -> None:
    cases = [
        ("[0/90]s", 0.0, "x"),
        ("[0/45/90]s", 0.0, "x"),
        ("[0/45/90]s", 0.0, "y"),
    ]
    for layup, shift, load_dir in cases:
        print(f"Solving {layup} (gap {shift:.2f}, {load_dir}-tension) ...")
        visualize_layup(layup, LAYUPS[layup], shift, load_dir)


if __name__ == "__main__":
    main()
