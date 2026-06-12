"""Assemble + solve driver for the 2D spring-network off-axis solver."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

from springshear.spring2d.assemble2d import assemble, build_interlaminar_elements
from springshear.spring2d.bc2d import build_tension_bcs
from springshear.spring2d.mesh2d import Mesh2D, build_mesh
from springshear.spring2d.params2d import Params2D


def _apply_dirichlet(K, f, prescribed: dict[int, float]):
    n = f.shape[0]
    u_full = np.zeros(n)
    for dof, val in prescribed.items():
        u_full[dof] = val

    pres = np.array(sorted(prescribed.keys()), dtype=int)
    is_pres = np.zeros(n, dtype=bool)
    is_pres[pres] = True
    free = np.where(~is_pres)[0]

    K_ff = K[free][:, free]
    K_fp = K[free][:, pres]
    f_eff = f[free] - K_fp @ u_full[pres]
    return K_ff, f_eff, free, u_full


def solve2d(params: Params2D, mesh: Mesh2D | None = None):
    """Return (mesh, u) where u is the full (2*N,) displacement vector."""
    if mesh is None:
        mesh = build_mesh(params)
        mesh.elements.extend(build_interlaminar_elements(params, mesh))

    K = assemble(params, mesh)
    f = np.zeros(2 * mesh.n_nodes)

    prescribed = build_tension_bcs(params, mesh)
    K_ff, f_eff, free, u_full = _apply_dirichlet(K, f, prescribed)
    u_free = spsolve(K_ff.tocsc(), f_eff)
    u_full[free] = u_free
    return mesh, u_full
