"""Apparent-modulus comparison across layups at the bulk-converged RVE size.

Isolates how stacking and through-thickness gap stagger affect gap tolerance,
using the apparent laminate modulus E_eff = R/(eps0*A). Cases:

    [0] no-gap   continuous unidirectional ply (knockdown-free anchor)
    [0]          single ply WITH severed gaps (no through-thickness bridging)
    [0]4 aligned 4 stacked 0deg plies, gaps stacked through thickness
    [0]4 stagg.  4 stacked 0deg plies, gaps staggered through thickness
    [0/90]s      cross-ply
    [0/45/90]s   quasi-isotropic (symmetric)
    [0/45/90]    quasi-isotropic (unsymmetric)

A common long gauge is used so every case is grip-converged.

Run from the repo root:
    $env:MPLBACKEND="Agg"; python examples/offaxis_2d/compare_stiffness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from springshear.spring2d.metrics2d import effective_modulus, evaluate2d  # noqa: E402
from springshear.spring2d.params2d import Params2D  # noqa: E402
from springshear.spring2d.solve2d import solve2d  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# Long gauge so every layup is grip-converged (UD needs ~14 periods).
NX_TOWS, NY_TOWS = 14, 16

# (label, angles, per-ply longitudinal gap shift, continuous?)
CASES = [
    ("[0] no-gap", [0.0], None, True),
    ("[0]", [0.0], None, False),
    ("[0]4 aligned", [0.0] * 4, [0.0, 0.0, 0.0, 0.0], False),
    ("[0]4 stagg.", [0.0] * 4, [0.0, 0.25, 0.5, 0.75], False),
    ("[0/90]s", [0.0, 90.0, 90.0, 0.0], None, False),
    ("[0/45/90]s", [0.0, 45.0, -45.0, 90.0], None, False),
    ("[0/45/90]", [0.0, 45.0, 90.0], None, False),
]


def run(angles: list[float], gap_shifts, continuous: bool, load_dir: str) -> dict:
    base = Params2D(ply_angles=angles)
    kwargs = dict(
        Lx=NX_TOWS * base.period,
        Ly=NY_TOWS * base.pitch,
        ply_angles=angles,
        load_dir=load_dir,
    )
    if gap_shifts is not None:
        kwargs["ply_gap_shifts"] = gap_shifts
    if continuous:
        kwargs["L_fiber"] = 10.0  # >> Lx => no gaps within the RVE
    params = Params2D(**kwargs)
    mesh, u = solve2d(params)
    return {
        "E_eff": effective_modulus(params, mesh, u),
        "cv": evaluate2d(params, mesh, u)["stress_cv"],
    }


def main() -> None:
    print(f"Apparent modulus comparison at {NX_TOWS}x{NY_TOWS} tows (x-tension)\n")
    print(f"{'layup':<14}{'E_eff (GPa)':>14}{'load-dir CV':>14}")
    print("-" * 42)
    names, e_vals, cv_vals = [], [], []
    for name, angles, gap_shifts, continuous in CASES:
        r = run(angles, gap_shifts, continuous, "x")
        names.append(name)
        e_vals.append(r["E_eff"] / 1e9)
        cv_vals.append(r["cv"])
        cv_txt = f"{r['cv']:.4f}" if r["cv"] == r["cv"] else "  n/a"
        print(f"{name:<14}{r['E_eff']/1e9:>14.2f}{cv_txt:>14}")

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    colors = ["0.6", "tab:red", "tab:orange", "tab:green", "tab:blue", "tab:blue", "tab:blue"]
    bars = ax.bar(names, e_vals, color=colors[: len(names)], edgecolor="black")
    ax.set_ylabel("apparent modulus $E_{eff}$ (GPa)")
    ax.set_title(f"Apparent laminate modulus by layup (x-tension, {NX_TOWS}x{NY_TOWS} tows)")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=20)
    for b, v in zip(bars, e_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=9)
    out_path = OUT_DIR / "compare_stiffness.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_path}")

    e_cont = e_vals[names.index("[0] no-gap")]
    e_uni = e_vals[names.index("[0]")]
    e_al = e_vals[names.index("[0]4 aligned")]
    e_st = e_vals[names.index("[0]4 stagg.")]
    print("\n" + "=" * 64)
    print(f"Gap knockdown (single ply): {e_cont:.1f} -> {e_uni:.1f} GPa "
          f"({(e_uni/e_cont - 1)*100:+.0f}%) with severed gaps and no bridging.")
    print(f"Through-thickness stagger: [0]4 aligned = {e_al:.1f} GPa vs "
          f"[0]4 staggered = {e_st:.1f} GPa ({(e_st/e_al - 1)*100:+.1f}%).")
    print("=> Aligned gaps give NO bridging; staggering gaps through the")
    print("   thickness lets neighbouring plies carry past a severed tow.")
    print("=" * 64)


if __name__ == "__main__":
    main()
