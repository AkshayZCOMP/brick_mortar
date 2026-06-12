from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

from springshear.assembly.assemble_stack import assemble_model
from springshear.bc.dirichlet import apply_dirichlet_elimination, build_fixed_bcs
from springshear.bc.periodic import apply_periodic_bc, expand_periodic_solution
from springshear.geometry.mesh import build_mesh
from springshear.params import Params


def solve(params: Params, x: np.ndarray | None = None):
    if x is None:
        x = build_mesh(params.L, params.dx)

    K, f, elems, dof = assemble_model(params, x)
    K = K.tocsr()
    n = dof.total_dofs()

    if params.bc_mode == "periodic" and params.periodic_x:
        K_ff, f_eff, free_dofs, pairs, old_to_new = apply_periodic_bc(K, f, params, dof)
        offset = params.eps0 * params.L
        mid_node = dof.n_nodes // 2
        pin_full = dof.dof_tow(0, 0, mid_node)
        pin_red = int(old_to_new[pin_full])

        u_red = np.zeros(K.shape[0] - len(pairs))
        u_red[pin_red] = 0.0
        u_free = spsolve(K_ff, f_eff)
        u_red[free_dofs] = u_free
        u_full = expand_periodic_solution(u_red, pairs, old_to_new, n, offset)
    else:
        prescribed = build_fixed_bcs(params, dof)
        K_ff, f_eff, free_dofs, u_full = apply_dirichlet_elimination(K, f, prescribed)
        u_free = spsolve(K_ff, f_eff)
        u_full[free_dofs] = u_free

    return x, u_full, elems, dof
