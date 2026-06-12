from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from springshear.assembly.dof import DofMap
from springshear.bc.dirichlet import apply_dirichlet_elimination
from springshear.params import Params


def build_periodic_pairs(dof: DofMap, params: Params) -> list[tuple[int, int]]:
    """Master (left x=0) / slave (right x=L) DOF pairs for axial periodicity."""
    n_nodes = dof.n_nodes
    left_i = 0
    right_i = n_nodes - 1
    pairs: list[tuple[int, int]] = []

    for ply in range(params.n_plies):
        for row in range(params.n_rows):
            pairs.append((dof.dof_tow(ply, row, left_i), dof.dof_tow(ply, row, right_i)))

        for r_int in range(dof.n_intf_y()):
            pairs.append((dof.dof_intf_y(ply, r_int, left_i), dof.dof_intf_y(ply, r_int, right_i)))

    for boundary in range(dof.n_boundaries()):
        for row in range(params.n_rows):
            pairs.append(
                (dof.dof_inter_ply(boundary, row, left_i), dof.dof_inter_ply(boundary, row, right_i))
            )

    return pairs


def _build_reduction_map(n: int, slaves: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """Map full DOF indices to reduced indices; slaves map to -1."""
    keep = np.array([i for i in range(n) if i not in slaves], dtype=int)
    old_to_new = -np.ones(n, dtype=int)
    old_to_new[keep] = np.arange(len(keep), dtype=int)
    return keep, old_to_new


def eliminate_periodic_pairs(
    K: csr_matrix,
    f: np.ndarray,
    pairs: list[tuple[int, int]],
    offset: float,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """
    Enforce u_slave = u_master + offset, eliminating slave DOFs from the system.
    """
    K = K.tolil()
    f = np.array(f, dtype=float).copy()
    slaves = {s for _, s in pairs}

    for master, slave in pairs:
        col_s = np.asarray(K[:, slave].todense()).ravel()
        row_s = np.asarray(K[slave, :].todense()).ravel()
        f -= col_s * offset
        K[:, master] = K[:, master] + K[:, slave]
        K[master, :] = K[master, :] + row_s

    keep, old_to_new = _build_reduction_map(K.shape[0], slaves)
    K_red = K[keep][:, keep].tocsr()
    f_red = f[keep]
    return K_red, f_red, old_to_new


def expand_periodic_solution(
    u_red: np.ndarray,
    pairs: list[tuple[int, int]],
    old_to_new: np.ndarray,
    n_full: int,
    offset: float,
) -> np.ndarray:
    u_full = np.zeros(n_full)
    for old_i in range(n_full):
        new_i = old_to_new[old_i]
        if new_i >= 0:
            u_full[old_i] = u_red[new_i]
    for master, slave in pairs:
        u_full[slave] = u_full[master] + offset
    return u_full


def apply_periodic_bc(
    K: csr_matrix,
    f: np.ndarray,
    params: Params,
    dof: DofMap,
) -> tuple[csr_matrix, np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    """
    Apply axial periodicity and pin one DOF.
    Returns K_ff, f_eff, free_dofs, pairs, old_to_new (after periodic reduction).
    """
    pairs = build_periodic_pairs(dof, params)
    offset = params.eps0 * params.L
    K_red, f_red, old_to_new = eliminate_periodic_pairs(K, f, pairs, offset)

    # Pin a mid-span tow DOF (not at x=0) to remove rigid-body mode without a
    # boundary spike in the first axial element.
    mid_node = dof.n_nodes // 2
    pin_full = dof.dof_tow(0, 0, mid_node)
    pin_red = int(old_to_new[pin_full])
    K_ff, f_eff, free_dofs, _ = apply_dirichlet_elimination(K_red, f_red, {pin_red: 0.0})
    return K_ff, f_eff, free_dofs, pairs, old_to_new
