"""RVE size convergence study for the 2D off-axis solver (fixed-grip BCs).

Because the off-axis plies use fixed-grip (non-periodic) boundary conditions,
the modelled domain must be large enough that the grip boundary layer (loaded
faces) and free-edge effects (transverse faces) do not contaminate the bulk
response. This study sizes the RVE in *tow counts*:

    Lx = Nx * period   (Nx brick-periods along the load-direction tow)
    Ly = Ny * pitch    (Ny tow strips across the width)

It sweeps Nx (width held fixed) and Ny (length held fixed), tracking two
convergence indicators measured away from the grips:

    * load-direction tow-stress CV (central 50% of the span)
    * apparent laminate modulus E_eff = R / (eps0 * A) from the grip reaction

and recommends the smallest (Nx, Ny) whose metrics are within `TOL` of the
largest case tested.

Run from the repo root:

    python examples/offaxis_2d/convergence.py
    $env:MPLBACKEND="Agg"; python examples/offaxis_2d/convergence.py   # headless
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from springshear.spring2d.metrics2d import (  # noqa: E402
    effective_modulus,
    matrix_shear_stresses,
    stress_cv,
    tow_axial_stresses,
)
from springshear.spring2d.params2d import Params2D  # noqa: E402
from springshear.spring2d.solve2d import solve2d  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# --- study configuration ---
# Presets selectable from the command line, e.g.
#   python examples/offaxis_2d/convergence.py uni
#   python examples/offaxis_2d/convergence.py qi
PRESETS = {
    "uni": ("[0]", [0.0], "x"),
    "0_90s": ("[0/90]s", [0.0, 90.0, 90.0, 0.0], "x"),
    "qi": ("[0/45/90]s", [0.0, 45.0, -45.0, 90.0], "x"),
}

LAYUP_NAME = "[0/45/90]s"
LAYUP_ANGLES = [0.0, 45.0, -45.0, 90.0]
LOAD_DIR = "x"
TOL = 0.02  # relative tolerance for "converged"

NX_SWEEP = [2, 4, 6, 8, 10, 14, 18, 24]
NY_SWEEP = [4, 6, 8, 10, 12, 16, 20]
NY_FIXED = 12   # strips held fixed while sweeping Nx
NX_FIXED = 14   # periods held fixed while sweeping Ny (long enough to be grip-free)


def solve_at(nx: int, ny: int) -> dict:
    base = Params2D(ply_angles=LAYUP_ANGLES, load_dir=LOAD_DIR)
    params = Params2D(
        Lx=nx * base.period,
        Ly=ny * base.pitch,
        ply_angles=LAYUP_ANGLES,
        load_dir=LOAD_DIR,
    )
    t0 = time.time()
    mesh, u = solve2d(params)
    dt = time.time() - t0

    _, span = (0, params.Lx) if LOAD_DIR == "x" else (1, params.Ly)
    sig = tow_axial_stresses(params, mesh, u, load_direction_only=True, grip_margin=0.25 * span)
    tau = matrix_shear_stresses(params, mesh, u)
    return {
        "nx": nx,
        "ny": ny,
        "n_nodes": mesh.n_nodes,
        "cv": stress_cv(sig),
        "mean_sig": float(np.mean(sig)) if sig.size else float("nan"),
        "E_eff": effective_modulus(params, mesh, u),
        "tau_max": float(np.max(np.abs(tau))) if tau.size else float("nan"),
        "n_tows": int(sig.size),
        "time_s": dt,
    }


def _print_sweep(label: str, rows: list[dict], var: str) -> None:
    print(f"\n=== {LAYUP_NAME} | {LOAD_DIR}-tension | sweep {label} ===")
    hdr = f"{var:>4} {'nodes':>8} {'CV':>9} {'mean_sig(GPa)':>14} {'E_eff(GPa)':>12} {'tau_max(MPa)':>13} {'t(s)':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r[var]:>4} {r['n_nodes']:>8} {r['cv']:>9.4f} "
            f"{r['mean_sig']/1e9:>14.4f} {r['E_eff']/1e9:>12.3f} "
            f"{r['tau_max']/1e6:>13.1f} {r['time_s']:>7.2f}"
        )


def _converged_index(rows: list[dict], keys=("cv", "E_eff")) -> tuple[int, bool]:
    """Convergence by *successive* relative change.

    Returns (index, fully_converged). `index` is the first size beyond which all
    successive changes in every key stay within TOL. `fully_converged` is False
    if even the final refinement step still exceeds TOL (range too small).
    """
    last_big = 0
    for i in range(1, len(rows)):
        for k in keys:
            prev, cur = rows[i - 1][k], rows[i][k]
            if prev and abs(cur - prev) / abs(prev) > TOL:
                last_big = i
                break
    fully = last_big < len(rows) - 1
    return last_big, fully


def _print_deltas(rows: list[dict], var: str) -> None:
    print(f"  successive change ({var}):")
    for i in range(1, len(rows)):
        d_cv = abs(rows[i]["cv"] - rows[i - 1]["cv"]) / abs(rows[i - 1]["cv"])
        d_e = abs(rows[i]["E_eff"] - rows[i - 1]["E_eff"]) / abs(rows[i - 1]["E_eff"])
        flag = "  <-- both < TOL" if (d_cv <= TOL and d_e <= TOL) else ""
        print(f"    {rows[i-1][var]:>3}->{rows[i][var]:<3}  dCV={d_cv*100:5.1f}%  dE={d_e*100:5.1f}%{flag}")


def plot(nx_rows: list[dict], ny_rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")

    def panel(ax, rows, var, key, ylabel, scale, marker):
        xs = [r[var] for r in rows]
        ys = [r[key] / scale for r in rows]
        ax.plot(xs, ys, marker + "-", color="tab:blue")
        ax.set_xlabel(f"{'Nx (brick-periods, x)' if var == 'nx' else 'Ny (tow strips, y)'}")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    panel(axes[0, 0], nx_rows, "nx", "cv", "load-dir tow-stress CV", 1.0, "o")
    panel(axes[0, 1], nx_rows, "nx", "E_eff", "E_eff (GPa)", 1e9, "o")
    panel(axes[1, 0], ny_rows, "ny", "cv", "load-dir tow-stress CV", 1.0, "s")
    panel(axes[1, 1], ny_rows, "ny", "E_eff", "E_eff (GPa)", 1e9, "s")

    i_nx, _ = _converged_index(nx_rows)
    i_ny, _ = _converged_index(ny_rows)
    for ax in (axes[0, 0], axes[0, 1]):
        ax.axvline(nx_rows[i_nx]["nx"], color="tab:red", ls="--", alpha=0.7)
    for ax in (axes[1, 0], axes[1, 1]):
        ax.axvline(ny_rows[i_ny]["ny"], color="tab:red", ls="--", alpha=0.7)

    fig.suptitle(
        f"RVE size convergence: {LAYUP_NAME}, {LOAD_DIR}-tension "
        f"(top: vary Nx @ Ny={NY_FIXED}; bottom: vary Ny @ Nx={NX_FIXED}; "
        f"red = converged within {TOL:.0%})",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")


def main() -> None:
    global LAYUP_NAME, LAYUP_ANGLES, LOAD_DIR
    preset = sys.argv[1] if len(sys.argv) > 1 else "qi"
    if preset not in PRESETS:
        raise SystemExit(f"Unknown preset {preset!r}; choose from {list(PRESETS)}")
    LAYUP_NAME, LAYUP_ANGLES, LOAD_DIR = PRESETS[preset]

    print(f"RVE convergence study for {LAYUP_NAME} ({LOAD_DIR}-tension), TOL={TOL:.0%}")

    nx_rows = [solve_at(nx, NY_FIXED) for nx in NX_SWEEP]
    _print_sweep(f"Nx (Ny={NY_FIXED} fixed)", nx_rows, "nx")
    _print_deltas(nx_rows, "nx")

    ny_rows = [solve_at(NX_FIXED, ny) for ny in NY_SWEEP]
    _print_sweep(f"Ny (Nx={NX_FIXED} fixed)", ny_rows, "ny")
    _print_deltas(ny_rows, "ny")

    i_nx, nx_ok = _converged_index(nx_rows)
    i_ny, ny_ok = _converged_index(ny_rows)
    nx_c = nx_rows[i_nx]["nx"]
    ny_c = ny_rows[i_ny]["ny"]

    base = Params2D(ply_angles=LAYUP_ANGLES)
    print("\n" + "=" * 64)
    print(f"RECOMMENDED PLY SIZE (successive change within {TOL:.0%}):")
    nx_note = "" if nx_ok else "  [NOT CONVERGED - extend NX_SWEEP]"
    ny_note = "" if ny_ok else "  [NOT CONVERGED - extend NY_SWEEP]"
    print(f"  Nx = {nx_c} brick-periods along load (Lx = {nx_c * base.period * 1e3:.1f} mm){nx_note}")
    print(f"  Ny = {ny_c} tow strips across width  (Ly = {ny_c * base.pitch * 1e3:.1f} mm){ny_note}")
    if nx_ok and ny_ok:
        print(f"  => model at least {nx_c} x {ny_c} tows for bulk-converged off-axis results")
    else:
        print("  => range insufficient in flagged direction(s); enlarge and re-run")
    print("=" * 64)

    slug = LAYUP_NAME.replace("[", "").replace("]", "").replace("/", "_")
    plot(nx_rows, ny_rows, OUT_DIR / f"convergence_{slug}.png")


if __name__ == "__main__":
    main()
