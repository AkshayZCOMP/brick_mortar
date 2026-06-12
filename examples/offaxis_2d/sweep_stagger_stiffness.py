"""Through-thickness gap-stagger sweep with STIFFNESS as the objective.

The compare_stiffness.py study showed (at two points) that staggering resin gaps
through the thickness dramatically recovers stiffness lost to severed tows. This
script turns that into a continuous curve: it sweeps a single through-thickness
stagger parameter s and records the apparent modulus E_eff (and load-direction
stress CV) at each step.

Stagger model: ply i receives a longitudinal gap shift (i * s) mod 1 (fraction
of the brick period). s = 0 stacks all gaps; s = 1/N spreads the N plies' gaps
evenly through the thickness (maximal bridging); by symmetry E_eff(s) = E_eff(1-s).

Run from the repo root:
    $env:MPLBACKEND="Agg"; python examples/offaxis_2d/sweep_stagger_stiffness.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from springshear.spring2d.metrics2d import effective_modulus, evaluate2d  # noqa: E402
from springshear.spring2d.params2d import Params2D  # noqa: E402
from springshear.spring2d.solve2d import solve2d  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# Long gauge so each point is grip-converged.
NX_TOWS, NY_TOWS = 12, 12
S_VALUES = np.round(np.linspace(0.0, 1.0, 21), 3)

STACKS = {
    "[0]4": [0.0] * 4,
    "[0]8": [0.0] * 8,
    "[0/45/90]s": [0.0, 45.0, -45.0, 90.0],
}


def run_point(angles: list[float], s: float) -> dict:
    n = len(angles)
    base = Params2D(ply_angles=angles)
    params = Params2D(
        Lx=NX_TOWS * base.period,
        Ly=NY_TOWS * base.pitch,
        ply_angles=angles,
        ply_gap_shifts=[(i * s) % 1.0 for i in range(n)],
        load_dir="x",
    )
    mesh, u = solve2d(params)
    return {
        "E_eff": effective_modulus(params, mesh, u) / 1e9,
        "cv": evaluate2d(params, mesh, u)["stress_cv"],
    }


def main() -> None:
    print(f"Through-thickness stagger sweep (stiffness objective) at {NX_TOWS}x{NY_TOWS} tows\n")
    results: dict[str, dict] = {}
    rows: list[dict] = []
    for name, angles in STACKS.items():
        n = len(angles)
        e_list, cv_list = [], []
        print(f"--- {name} (N={n}) ---")
        for s in S_VALUES:
            r = run_point(angles, float(s))
            e_list.append(r["E_eff"])
            cv_list.append(r["cv"])
            rows.append({"stack": name, "s": s, "E_eff_GPa": round(r["E_eff"], 3),
                         "cv": round(r["cv"], 4)})
        e_arr = np.array(e_list)
        i_best = int(np.argmax(e_arr))
        e0 = e_arr[0]
        results[name] = {"s": S_VALUES, "E": e_arr, "cv": np.array(cv_list),
                         "i_best": i_best, "n": n, "e0": e0}
        print(f"  aligned (s=0): E_eff={e0:.1f} GPa")
        print(f"  best: s={S_VALUES[i_best]:.2f}  E_eff={e_arr[i_best]:.1f} GPa "
              f"({(e_arr[i_best]/e0 - 1)*100:+.0f}%)   even-spacing 1/N={1.0/n:.3f}\n")

    csv_path = OUT_DIR / "sweep_stagger_stiffness.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stack", "s", "E_eff_GPa", "cv"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), layout="constrained", sharex=True)
    colors = {"[0]4": "tab:red", "[0]8": "tab:purple", "[0/45/90]s": "tab:blue"}
    for name, d in results.items():
        c = colors.get(name, "tab:gray")
        ax1.plot(d["s"], d["E"], "o-", color=c, label=name)
        ax1.plot(d["s"][d["i_best"]], d["E"][d["i_best"]], "*", color=c, ms=16,
                 markeredgecolor="black")
        ax1.axvline(1.0 / d["n"], color=c, ls=":", alpha=0.5)
        ax2.plot(d["s"], d["cv"], "s-", color=c, label=name)
    ax1.set_ylabel("apparent modulus $E_{eff}$ (GPa)")
    ax1.set_title(f"Stiffness vs through-thickness gap stagger ({NX_TOWS}x{NY_TOWS} tows, x-tension)\n"
                  "star = max; dotted = even 1/N spacing")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.set_ylabel("load-direction tow-stress CV")
    ax2.set_xlabel("through-thickness stagger  s  (per-layer gap shift, fraction of period)")
    ax2.grid(alpha=0.3)
    ax2.legend()
    out_path = OUT_DIR / "sweep_stagger_stiffness.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    print("\n" + "=" * 64)
    for name, d in results.items():
        s_best = d["s"][d["i_best"]]
        print(f"{name:<12} optimum stagger s={s_best:.2f} (even 1/N={1.0/d['n']:.3f}): "
              f"E_eff {d['e0']:.1f} -> {d['E'][d['i_best']]:.1f} GPa "
              f"({(d['E'][d['i_best']]/d['e0'] - 1)*100:+.0f}%)")
    print("=> Stiffness is maximized near even 1/N through-thickness gap spacing.")
    print("=" * 64)


if __name__ == "__main__":
    main()
