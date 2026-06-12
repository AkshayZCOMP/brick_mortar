from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix

from springshear.assembly.assemble_ply import assemble_ply
from springshear.assembly.dof import DofMap
from springshear.elements.springs import add_2node_spring, k_axial, k_shear
from springshear.geometry.staggering import inter_ply_x_weight, ply_y_shift, row_overlap, segment_xmid
from springshear.params import Params


def _inter_ply_weights(
    params: Params,
    ply_lower: int,
    ply_upper: int,
    ipm_row: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Overlap weights from each ply tow row to an inter-ply matrix row index."""
    y_lo = ply_y_shift(params, ply_lower)
    y_up = ply_y_shift(params, ply_upper)
    lo_weights: list[tuple[int, float]] = []
    up_weights: list[tuple[int, float]] = []

    lo_totals = [
        sum(
            row_overlap(row, y_lo, ipm, 0.0, params.row_width, params.row_pitch)
            for ipm in range(params.n_rows)
        )
        for row in range(params.n_rows)
    ]
    up_totals = [
        sum(
            row_overlap(row, y_up, ipm, 0.0, params.row_width, params.row_pitch)
            for ipm in range(params.n_rows)
        )
        for row in range(params.n_rows)
    ]

    for row in range(params.n_rows):
        w_lo = row_overlap(row, y_lo, ipm_row, 0.0, params.row_width, params.row_pitch)
        if w_lo > 0 and lo_totals[row] > 0:
            lo_weights.append((row, w_lo / lo_totals[row]))
        w_up = row_overlap(row, y_up, ipm_row, 0.0, params.row_width, params.row_pitch)
        if w_up > 0 and up_totals[row] > 0:
            up_weights.append((row, w_up / up_totals[row]))

    return lo_weights, up_weights


def assemble_stack(params: Params, x: np.ndarray) -> tuple[lil_matrix, np.ndarray, list, DofMap]:
    dof = DofMap(params, len(x))
    K = lil_matrix((dof.total_dofs(), dof.total_dofs()))
    f = np.zeros(dof.total_dofs())
    elems: list = []

    for ply in range(params.n_plies):
        K, elems = assemble_ply(params, x, dof, ply=ply, K=K, elems=elems)

    if params.n_plies <= 1 and not params.periodic_z:
        return K, f, elems, dof

    ksh = k_shear(params.G_m, params.b_eff, params.t_eff, params.dx)
    k_tm = 2.0 * ksh
    k_m_ax = k_axial(params.E_m, params.A_m, params.dx)
    n_nodes = len(x)

    for boundary in range(dof.n_boundaries()):
        ply_lo, ply_hi = dof.boundary_plies(boundary)
        for ipm_row in range(params.n_rows):
            lo_w, hi_w = _inter_ply_weights(params, ply_lo, ply_hi, ipm_row)
            lo_rows = [row for row, _ in lo_w]
            hi_rows = [row for row, _ in hi_w]
            for i in range(n_nodes):
                ipm = dof.dof_inter_ply(boundary, ipm_row, i)
                xmid = segment_xmid(x, i)
                for row, w_y in lo_w:
                    w_x = inter_ply_x_weight(params, ply_lo, row, xmid, ply_hi, hi_rows)
                    k_eff = k_tm * w_y * w_x
                    if k_eff <= 0:
                        continue
                    tow = dof.dof_tow(ply_lo, row, i)
                    add_2node_spring(K, tow, ipm, k_eff)
                    elems.append(
                        {
                            "etype": "ip_tm_shear",
                            "boundary": boundary,
                            "ply": ply_lo,
                            "row": row,
                            "ipm_row": ipm_row,
                            "i": i,
                            "a": tow,
                            "m": ipm,
                            "k": k_eff,
                            "w_x": w_x,
                            "w_y": w_y,
                        }
                    )
                for row, w_y in hi_w:
                    w_x = inter_ply_x_weight(params, ply_hi, row, xmid, ply_lo, lo_rows)
                    k_eff = k_tm * w_y * w_x
                    if k_eff <= 0:
                        continue
                    tow = dof.dof_tow(ply_hi, row, i)
                    add_2node_spring(K, tow, ipm, k_eff)
                    elems.append(
                        {
                            "etype": "ip_tm_shear",
                            "boundary": boundary,
                            "ply": ply_hi,
                            "row": row,
                            "ipm_row": ipm_row,
                            "i": i,
                            "a": tow,
                            "m": ipm,
                            "k": k_eff,
                            "w_x": w_x,
                            "w_y": w_y,
                        }
                    )

        for ipm_row in range(params.n_rows):
            for i in range(n_nodes - 1):
                ipm0 = dof.dof_inter_ply(boundary, ipm_row, i)
                ipm1 = dof.dof_inter_ply(boundary, ipm_row, i + 1)
                add_2node_spring(K, ipm0, ipm1, k_m_ax)
                elems.append(
                    {
                        "etype": "ip_m_axial",
                        "boundary": boundary,
                        "ipm_row": ipm_row,
                        "i": i,
                        "a": ipm0,
                        "b": ipm1,
                        "k": k_m_ax,
                        "A": params.A_m,
                    }
                )

    return K, f, elems, dof


def assemble_model(params: Params, x: np.ndarray) -> tuple[lil_matrix, np.ndarray, list, DofMap]:
    """Assemble single ply or multi-ply stack depending on params.n_plies."""
    if params.n_plies == 1 and not params.periodic_z:
        dof = DofMap(params, len(x))
        K = lil_matrix((dof.total_dofs(), dof.total_dofs()))
        f = np.zeros(dof.total_dofs())
        K, elems = assemble_ply(params, x, dof, ply=0, K=K, elems=[])
        return K, f, elems, dof
    return assemble_stack(params, x)
