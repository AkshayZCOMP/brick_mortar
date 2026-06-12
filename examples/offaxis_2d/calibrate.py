"""Calibrate the 2D spring-shear network against continuum FEA.

The spring network has two physically uncertain stiffness groups: the matrix
shear-lag springs (along-tow transverse shear + inter-strip shear) that let load
detour around a severed tow, and the inter-strip normal springs that set the
transverse modulus. Their effective magnitudes depend on an idealized resin
ligament geometry, so we expose two dimensionless multipliers,
``shear_lag_factor`` and ``transverse_factor`` (both =1.0 from first principles),
and *calibrate* them.

Calibration targets (defensible, ratio-based so the absolute-modulus idealization
cancels), measured with the independent Q4 continuum FE solver:
  * shear_lag_factor  -> matches the 0deg single-ply GAP KNOCKDOWN ratio
                         E_gapped / E_no-gap (shear-lag / bridging fidelity).
  * transverse_factor -> matches the 90deg/0deg transverse-to-axial modulus ratio.
We then VALIDATE (no further tuning) that the calibrated network reproduces the
FE 45deg off-axis modulus and the through-thickness stagger recovery of a [0]4
stack.

Run from the repo root:
    $env:MPLBACKEND="Agg"; python examples/offaxis_2d/calibrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from springshear.spring2d.metrics2d import effective_modulus  # noqa: E402
from springshear.spring2d.params2d import Params2D  # noqa: E402
from springshear.spring2d.solve2d import solve2d  # noqa: E402

from fea_verify import (  # noqa: E402
    make_inplace_material,
    make_xz_stack_material,
    solve_planestress,
)

OUT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Spring-network apparent moduli (with calibration factor)
# --------------------------------------------------------------------------- #
def spring_inplane(angle: float, gapped: bool, Lx: float, Ly: float,
                   slf: float, tf: float = 1.0) -> float:
    kw = dict(Lx=Lx, Ly=Ly, ply_angles=[angle], load_dir="x",
              shear_lag_factor=slf, transverse_factor=tf)
    if not gapped:
        kw["L_fiber"] = 10.0  # one effectively continuous fibre period
    p = Params2D(**kw)
    mesh, u = solve2d(p)
    return effective_modulus(p, mesh, u) / 1e9


def spring_stack(s: float, Lx: float, Ly: float, slf: float, tf: float = 1.0) -> float:
    p = Params2D(Lx=Lx, Ly=Ly, ply_angles=[0.0] * 4,
                 ply_gap_shifts=[(i * s) % 1.0 for i in range(4)],
                 load_dir="x", shear_lag_factor=slf, transverse_factor=tf)
    mesh, u = solve2d(p)
    return effective_modulus(p, mesh, u) / 1e9


def main() -> None:
    base = Params2D()
    pitch, tow_width, period, L_fiber = base.pitch, base.tow_width, base.period, base.L_fiber
    t_ply, t_il = base.t_ply, base.interlaminar_thickness

    NX_T, NY_T = 6, 8
    Lx, Ly = NX_T * period, NY_T * pitch
    nx = max(120, int(Lx / 0.4e-3))
    ny = max(60, int(Ly / 0.4e-3))

    # --- FE calibration target: 0deg gap knockdown ratio ---
    fe_nogap = solve_planestress(nx, ny, Lx, Ly,
                                 make_inplace_material(0.0, pitch, tow_width, period, L_fiber, False)) / 1e9
    fe_gap = solve_planestress(nx, ny, Lx, Ly,
                               make_inplace_material(0.0, pitch, tow_width, period, L_fiber, True)) / 1e9
    fe_knock = fe_gap / fe_nogap
    print(f"FE 0deg: no-gap {fe_nogap:.1f} GPa, gapped {fe_gap:.1f} GPa, knockdown ratio {fe_knock:.3f}")

    # --- solve for shear_lag_factor so spring knockdown == FE knockdown ---
    sp_nogap = spring_inplane(0.0, False, Lx, Ly, 1.0)  # ~independent of slf

    def knock_residual(slf: float) -> float:
        return (spring_inplane(0.0, True, Lx, Ly, slf) / sp_nogap) - fe_knock

    def _solve_factor(residual, lo, hi, name):
        r_lo, r_hi = residual(lo), residual(hi)
        if r_lo * r_hi > 0:
            val = lo if abs(r_lo) < abs(r_hi) else hi
            print(f"WARNING: {name} not bracketed on [{lo},{hi}]; clamping to {val:.3f}")
            return val
        return brentq(residual, lo, hi, xtol=1e-3, rtol=1e-3)

    slf_cal = _solve_factor(knock_residual, 0.1, 30.0, "shear_lag_factor")
    print(f"Calibrated shear_lag_factor = {slf_cal:.3f}")

    # --- transverse_factor: match the FE 90deg/0deg transverse-to-axial ratio ---
    fe_90 = solve_planestress(nx, ny, Lx, Ly,
                              make_inplace_material(90.0, pitch, tow_width, period, L_fiber, True)) / 1e9
    target_trans_ratio = fe_90 / fe_nogap
    print(f"FE 90deg gapped {fe_90:.1f} GPa, transverse/axial ratio {target_trans_ratio:.3f}")

    def trans_residual(tf: float) -> float:
        return (spring_inplane(90.0, True, Lx, Ly, slf_cal, tf) / sp_nogap) - target_trans_ratio

    tf_cal = _solve_factor(trans_residual, 0.02, 5.0, "transverse_factor")
    print(f"Calibrated transverse_factor = {tf_cal:.3f}")

    # --- validation across angles + stack stagger, before vs after calibration ---
    cases = [("0deg gapped", 0.0, True), ("45deg gapped", 45.0, True), ("90deg gapped", 90.0, True)]
    rows = []
    for label, ang, gapped in cases:
        e_fe = solve_planestress(nx, ny, Lx, Ly,
                                 make_inplace_material(ang, pitch, tow_width, period, L_fiber, gapped)) / 1e9
        e_b = spring_inplane(ang, gapped, Lx, Ly, 1.0, 1.0)
        e_c = spring_inplane(ang, gapped, Lx, Ly, slf_cal, tf_cal)
        rows.append((label, e_fe, e_b, e_c))

    # stack stagger recovery ratio (staggered / aligned)
    H = 4 * t_ply + 3 * t_il
    nz = max(24, int(H / 0.025e-3))
    fe_al = solve_planestress(nx, nz, Lx, H, make_xz_stack_material(4, t_ply, t_il, period, L_fiber, 0.0)) / 1e9
    fe_st = solve_planestress(nx, nz, Lx, H, make_xz_stack_material(4, t_ply, t_il, period, L_fiber, 0.25)) / 1e9
    rec_fe = fe_st / fe_al
    rec_b = spring_stack(0.25, Lx, Ly, 1.0, 1.0) / spring_stack(0.0, Lx, Ly, 1.0, 1.0)
    rec_c = spring_stack(0.25, Lx, Ly, slf_cal, tf_cal) / spring_stack(0.0, Lx, Ly, slf_cal, tf_cal)

    print("\nValidation (E_eff, GPa) -- calibrated on 0deg knockdown + 90deg ratio:")
    print(f"{'case':<16}{'FE':>9}{'spring(slf=1)':>15}{'spring(cal)':>13}")
    print("-" * 53)
    for label, e_fe, e_b, e_c in rows:
        print(f"{label:<16}{e_fe:>9.1f}{e_b:>15.1f}{e_c:>13.1f}")
    print(f"{'stagger recovery':<16}{rec_fe:>9.2f}{rec_b:>15.2f}{rec_c:>13.2f}  (staggered/aligned)")

    # --- figure: knockdown/recovery, FE vs spring before/after calibration ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    labels = [r[0] for r in rows]
    x = np.arange(len(labels))
    w = 0.27
    ax1.bar(x - w, [r[1] for r in rows], w, label="continuum FEA", color="tab:orange")
    ax1.bar(x, [r[2] for r in rows], w, label="spring (uncalibrated)", color="0.6")
    ax1.bar(x + w, [r[3] for r in rows], w,
            label=f"spring (calibrated f={slf_cal:.2f}, t={tf_cal:.2f})", color="tab:blue")
    ax1.set_xticks(x, labels, rotation=12)
    ax1.set_ylabel("$E_{eff}$ (GPa)")
    ax1.set_title("In-plane single ply: calibration targets (0/90deg) + 45deg validation")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar([0], [rec_fe], 0.6, color="tab:orange", label="continuum FEA")
    ax2.bar([1], [rec_b], 0.6, color="0.6", label="spring (uncalibrated)")
    ax2.bar([2], [rec_c], 0.6, color="tab:blue",
            label=f"spring (calibrated f={slf_cal:.2f}, t={tf_cal:.2f})")
    for xi, v in [(0, rec_fe), (1, rec_b), (2, rec_c)]:
        ax2.text(xi, v + 0.04, f"x{v:.2f}", ha="center", fontsize=10)
    ax2.set_xticks([0, 1, 2], ["FEA", "spring\n(uncal.)", "spring\n(cal.)"])
    ax2.set_ylabel("stagger recovery ratio (staggered / aligned)")
    ax2.axhline(1.0, color="k", lw=0.8, ls=":")
    ax2.set_title("[0]4 through-thickness stagger recovery (validation)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    out = OUT_DIR / "calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")
    print("\nUse Params2D(..., shear_lag_factor={:.3f}, transverse_factor={:.3f}) "
          "for calibrated runs.".format(slf_cal, tf_cal))


if __name__ == "__main__":
    main()
