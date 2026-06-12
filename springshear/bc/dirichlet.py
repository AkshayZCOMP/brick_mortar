from __future__ import annotations

import numpy as np

from springshear.assembly.dof import DofMap
from springshear.params import Params


def apply_dirichlet_elimination(K, f, prescribed: dict[int, float]):
    n = f.shape[0]
    u_full = np.zeros(n)

    for dof, value in prescribed.items():
        u_full[dof] = value

    prescribed_dofs = np.array(sorted(prescribed.keys()))
    is_prescribed = np.zeros(n, dtype=bool)
    is_prescribed[prescribed_dofs] = True
    free_dofs = np.where(~is_prescribed)[0]

    K_ff = K[free_dofs][:, free_dofs]
    K_gp = K[free_dofs][:, prescribed_dofs]
    u_p = u_full[prescribed_dofs]
    f_f = f[free_dofs]
    f_eff = f_f - K_gp @ u_p
    return K_ff, f_eff, free_dofs, u_full


def build_fixed_bcs(params: Params, dof: DofMap) -> dict[int, float]:
    """Fixed-end BCs on all tow rows at x=0 and x=L for every ply."""
    n_nodes = dof.n_nodes
    left_i = 0
    right_i = n_nodes - 1
    u_right = params.eps0 * params.L
    prescribed: dict[int, float] = {}

    for ply in range(params.n_plies):
        for row in range(params.n_rows):
            prescribed[dof.dof_tow(ply, row, left_i)] = 0.0
            prescribed[dof.dof_tow(ply, row, right_i)] = u_right

    return prescribed
