"""Run single-ply 2D model with periodic BCs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from springshear.metrics.objectives import evaluate_objective
from springshear.params import Params
from springshear.post.plots import plot_ply_stresses
from springshear.solve.run import solve


def main():
    params = Params(
        n_plies=1,
        bc_mode="periodic",
        periodic_x=True,
        periodic_y=False,
        periodic_z=False,
    )
    x, u, elems, dof = solve(params)
    metrics = evaluate_objective(params, x, u, elems)
    print("Periodic 2D ply metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4e}" if isinstance(v, float) else f"  {k}: {v}")
    plot_ply_stresses(params, x, u, elems)


if __name__ == "__main__":
    main()
