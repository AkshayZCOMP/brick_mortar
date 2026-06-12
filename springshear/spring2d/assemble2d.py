"""Global stiffness assembly for the 2D spring-network solver (u_x, u_y per node)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

from springshear.spring2d.mesh2d import Mesh2D
from springshear.spring2d.params2d import Params2D


def _accumulate_directional(rows, cols, vals, a: int, b: int, dx: float, dy: float, k: float) -> None:
    """Add a spring penalizing relative displacement along (dx, dy)."""
    if k == 0.0:
        return
    dofs = (2 * a, 2 * a + 1, 2 * b, 2 * b + 1)
    grad = (-dx, -dy, dx, dy)
    for i in range(4):
        gi = grad[i]
        if gi == 0.0:
            continue
        for j in range(4):
            gj = grad[j]
            if gj == 0.0:
                continue
            rows.append(dofs[i])
            cols.append(dofs[j])
            vals.append(k * gi * gj)


def build_interlaminar_elements(params: Params2D, mesh: Mesh2D) -> list[dict]:
    """Couple adjacent plies by interlaminar in-plane shear (nearest-node)."""
    if params.n_plies < 2:
        return []
    t_il = params.interlaminar_thickness
    dA = params.dxi * params.pitch
    k_il = params.G_m * dA / t_il

    elements: list[dict] = []
    for lower in range(params.n_plies - 1):
        upper = lower + 1
        lo_idx = np.where(mesh.node_ply == lower)[0]
        up_idx = np.where(mesh.node_ply == upper)[0]
        if lo_idx.size == 0 or up_idx.size == 0:
            continue
        tree = cKDTree(mesh.nodes[lo_idx])
        _, nn = tree.query(mesh.nodes[up_idx], k=1)
        for u_local, l_local in zip(range(up_idx.size), nn):
            a = int(up_idx[u_local])
            b = int(lo_idx[l_local])
            # Isotropic in-plane interlaminar shear (x and y).
            elements.append(
                {"etype": "il_shear", "boundary": lower, "a": a, "b": b, "dx": 1.0, "dy": 0.0, "k": k_il}
            )
            elements.append(
                {"etype": "il_shear", "boundary": lower, "a": a, "b": b, "dx": 0.0, "dy": 1.0, "k": k_il}
            )
    return elements


def assemble(params: Params2D, mesh: Mesh2D) -> csr_matrix:
    n_dof = 2 * mesh.n_nodes
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for e in mesh.elements:
        _accumulate_directional(rows, cols, vals, e["a"], e["b"], e["dx"], e["dy"], e["k"])

    K = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    return K
