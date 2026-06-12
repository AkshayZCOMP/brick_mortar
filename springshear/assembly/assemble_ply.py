from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix

from springshear.assembly.dof import DofMap
from springshear.elements.springs import add_2node_spring, k_axial, k_shear
from springshear.geometry.staggering import (
    is_axial_ply,
    is_in_gap,
    is_in_gap_boundary,
    is_transverse_ply,
)
from springshear.params import Params


def _append_tow_segment(
    K: lil_matrix,
    elems: list,
    *,
    etype: str,
    ply: int,
    row: int,
    i: int,
    a: int,
    b: int,
    k: float,
    area: float,
    direction: str,
) -> None:
    add_2node_spring(K, a, b, k)
    elems.append(
        {
            "etype": etype,
            "ply": ply,
            "row": row,
            "i": i,
            "a": a,
            "b": b,
            "k": k,
            "A": area,
            "direction": direction,
        }
    )


def _assemble_axial_ply(
    params: Params,
    x: np.ndarray,
    dof: DofMap,
    ply: int,
    K: lil_matrix,
    elems: list,
) -> tuple[lil_matrix, list]:
    n_nodes = len(x)
    kax = k_axial(params.E_tow, params.A_tow, params.dx)
    ksh = k_shear(params.G_m, params.b_eff, params.t_eff, params.dx)
    k_tm = 2.0 * ksh
    A_m = params.A_m
    kbridge = k_axial(params.E_m, A_m, params.dx)
    k_m_ax = k_axial(params.E_m, A_m, params.dx)
    angle = params.ply_angle(ply)
    cos2 = float(np.cos(np.deg2rad(angle)) ** 2) if not is_axial_ply(angle) else 1.0
    kax_eff = kax * cos2
    kbridge_eff = kbridge * cos2

    n_intf = dof.n_intf_y()

    for row in range(params.n_rows):
        for i in range(n_nodes - 1):
            xmid = 0.5 * (x[i] + x[i + 1])
            # Gap-shift-aware membership (consistent with inter_ply_x_weight).
            inside = is_in_gap(params, ply, row, xmid)
            a = dof.dof_tow(ply, row, i)
            b = dof.dof_tow(ply, row, i + 1)
            if inside:
                _append_tow_segment(
                    K,
                    elems,
                    etype="tow_bridge",
                    ply=ply,
                    row=row,
                    i=i,
                    a=a,
                    b=b,
                    k=kbridge_eff,
                    area=A_m,
                    direction="x",
                )
            else:
                _append_tow_segment(
                    K,
                    elems,
                    etype="tow_axial",
                    ply=ply,
                    row=row,
                    i=i,
                    a=a,
                    b=b,
                    k=kax_eff,
                    area=params.A_tow,
                    direction="x",
                )

    for r_int in range(n_intf):
        row_b = (r_int + 1) % params.n_rows if params.periodic_y else r_int + 1
        for i in range(n_nodes):
            m = dof.dof_intf_y(ply, r_int, i)
            a = dof.dof_tow(ply, r_int, i)
            b = dof.dof_tow(ply, row_b, i)
            add_2node_spring(K, a, m, k_tm)
            add_2node_spring(K, b, m, k_tm)
            elems.append(
                {
                    "etype": "tm_shear",
                    "ply": ply,
                    "r_int": r_int,
                    "row": r_int,
                    "i": i,
                    "a": a,
                    "m": m,
                    "k": k_tm,
                }
            )
            elems.append(
                {
                    "etype": "tm_shear",
                    "ply": ply,
                    "r_int": r_int,
                    "row": row_b,
                    "i": i,
                    "a": b,
                    "m": m,
                    "k": k_tm,
                }
            )

    for r_int in range(n_intf):
        for i in range(n_nodes - 1):
            m0 = dof.dof_intf_y(ply, r_int, i)
            m1 = dof.dof_intf_y(ply, r_int, i + 1)
            add_2node_spring(K, m0, m1, k_m_ax)
            elems.append(
                {
                    "etype": "m_axial",
                    "ply": ply,
                    "r_int": r_int,
                    "i": i,
                    "a": m0,
                    "b": m1,
                    "k": k_m_ax,
                    "A": A_m,
                }
            )

    return K, elems


def _assemble_transverse_ply(
    params: Params,
    x: np.ndarray,
    dof: DofMap,
    ply: int,
    K: lil_matrix,
    elems: list,
) -> tuple[lil_matrix, list]:
    n_nodes = len(x)
    kax = k_axial(params.E_tow, params.A_tow, params.row_pitch)
    ksh = k_shear(params.G_m, params.b_eff, params.t_eff, params.dx)
    k_tm = 2.0 * ksh
    A_m = params.A_m
    kbridge = k_axial(params.E_m, A_m, params.row_pitch)
    k_m_ax = k_axial(params.E_m, A_m, params.dx)
    n_intf = dof.n_intf_y()

    for r_int in range(n_intf):
        row_b = (r_int + 1) % params.n_rows if params.periodic_y else r_int + 1
        for i in range(n_nodes):
            inside = is_in_gap_boundary(params, ply, r_int, float(x[i]))
            a = dof.dof_tow(ply, r_int, i)
            b = dof.dof_tow(ply, row_b, i)
            if inside:
                _append_tow_segment(
                    K,
                    elems,
                    etype="tow_bridge",
                    ply=ply,
                    row=r_int,
                    i=i,
                    a=a,
                    b=b,
                    k=kbridge,
                    area=A_m,
                    direction="y",
                )
            else:
                _append_tow_segment(
                    K,
                    elems,
                    etype="tow_axial",
                    ply=ply,
                    row=r_int,
                    i=i,
                    a=a,
                    b=b,
                    k=kax,
                    area=params.A_tow,
                    direction="y",
                )

    for r_int in range(n_intf):
        row_b = (r_int + 1) % params.n_rows if params.periodic_y else r_int + 1
        for i in range(n_nodes):
            m = dof.dof_intf_y(ply, r_int, i)
            a = dof.dof_tow(ply, r_int, i)
            b = dof.dof_tow(ply, row_b, i)
            add_2node_spring(K, a, m, k_tm)
            add_2node_spring(K, b, m, k_tm)
            elems.append(
                {
                    "etype": "tm_shear",
                    "ply": ply,
                    "r_int": r_int,
                    "row": r_int,
                    "i": i,
                    "a": a,
                    "m": m,
                    "k": k_tm,
                }
            )
            elems.append(
                {
                    "etype": "tm_shear",
                    "ply": ply,
                    "r_int": r_int,
                    "row": row_b,
                    "i": i,
                    "a": b,
                    "m": m,
                    "k": k_tm,
                }
            )

    for r_int in range(n_intf):
        for i in range(n_nodes - 1):
            m0 = dof.dof_intf_y(ply, r_int, i)
            m1 = dof.dof_intf_y(ply, r_int, i + 1)
            add_2node_spring(K, m0, m1, k_m_ax)
            elems.append(
                {
                    "etype": "m_axial",
                    "ply": ply,
                    "r_int": r_int,
                    "i": i,
                    "a": m0,
                    "b": m1,
                    "k": k_m_ax,
                    "A": A_m,
                }
            )

    return K, elems


def assemble_ply(
    params: Params,
    x: np.ndarray,
    dof: DofMap,
    ply: int = 0,
    K: lil_matrix | None = None,
    elems: list | None = None,
) -> tuple[lil_matrix, list]:
    """Assemble one ply into global K at the ply offset."""
    n_nodes = len(x)
    if K is None:
        K = lil_matrix((dof.total_dofs(), dof.total_dofs()))
    if elems is None:
        elems = []

    angle = params.ply_angle(ply)
    if is_transverse_ply(angle):
        return _assemble_transverse_ply(params, x, dof, ply, K, elems)
    return _assemble_axial_ply(params, x, dof, ply, K, elems)
