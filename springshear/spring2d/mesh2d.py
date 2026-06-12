"""Mesh and element generation for the 2D spring-network solver.

Each ply is discretized as a set of continuous tow strips at angle theta.
Nodes live on a regular (xi, eta) lattice aligned with the tow axis (xi along the
tow, eta perpendicular), clipped to the shared rectangular RVE footprint. Three
directional springs are generated:

  * tow       - axial tow stiffness along t = (cos t, sin t); SEVERED inside gaps
  * m_tshear  - matrix shear restraining transverse motion of the tow, along
                n = (-sin t, cos t), between consecutive along-tow nodes; present
                even across gaps (the resin ligament still resists transverse
                motion). This is the in-plane shear-lag restraint and is what
                lets load detour around a severed tow.
  * m_shear   - matrix shear between adjacent strips, along t
  * m_normal  - matrix transverse stiffness between adjacent strips, along n
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, cos, floor, radians, sin

import numpy as np

from springshear.spring2d.params2d import Params2D


@dataclass
class Mesh2D:
    nodes: np.ndarray              # (N, 2) x-y positions
    node_ply: np.ndarray           # (N,) ply index per node
    elements: list[dict] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)


def _in_gap(xi_mid: float, period: float, l_fiber: float, shift: float) -> bool:
    """Brick pattern: fiber occupies [0, L_fiber), gap occupies [L_fiber, period)."""
    r = (xi_mid - shift) % period
    return r >= l_fiber


def build_ply_mesh(
    params: Params2D,
    ply: int,
    node_offset: int,
) -> tuple[list[tuple[float, float]], dict[tuple[int, int], int], list[dict]]:
    theta = radians(params.ply_angle(ply))
    tx, ty = cos(theta), sin(theta)
    nx, ny = -sin(theta), cos(theta)

    Lx, Ly = params.Lx, params.Ly
    dxi, pitch = params.dxi, params.pitch

    # Transverse offset of this layer's tow-strip lattice (fraction of pitch).
    eta_shift = params.ply_transverse_shifts[ply] * pitch

    corners = [(0.0, 0.0), (Lx, 0.0), (0.0, Ly), (Lx, Ly)]
    xis = [cx * tx + cy * ty for cx, cy in corners]
    etas = [cx * nx + cy * ny for cx, cy in corners]
    j_lo, j_hi = floor(min(xis) / dxi) - 1, ceil(max(xis) / dxi) + 1
    k_lo = floor((min(etas) - eta_shift) / pitch) - 2
    k_hi = ceil((max(etas) - eta_shift) / pitch) + 2

    tol = 1e-12
    nodes: list[tuple[float, float]] = []
    index: dict[tuple[int, int], int] = {}
    for k in range(k_lo, k_hi + 1):
        eta = k * pitch + eta_shift
        for j in range(j_lo, j_hi + 1):
            xi = j * dxi
            x = xi * tx + eta * nx
            y = xi * ty + eta * ny
            if -tol <= x <= Lx + tol and -tol <= y <= Ly + tol:
                index[(j, k)] = node_offset + len(nodes)
                nodes.append((x, y))

    h_m = params.matrix_gap_thickness
    slf = params.shear_lag_factor
    k_ax = params.E_tow * params.A_tow / dxi
    k_tow_shear = slf * params.G_m * params.A_tow / dxi
    k_shear = slf * params.G_m * dxi * params.t_ply / h_m
    k_normal = params.transverse_factor * params.E_m * dxi * params.t_ply / h_m
    period = params.period
    gap_shift = params.ply_gap_shifts[ply] * period

    elements: list[dict] = []
    for (j, k), a in index.items():
        # Along the tow: axial (severed inside gaps) + matrix transverse-shear
        # restraint (always present; the resin carries transverse load and lets
        # the load path detour around a severed tow).
        b = index.get((j + 1, k))
        if b is not None:
            xi_mid = (j + 0.5) * dxi
            shift = gap_shift + k * params.strip_stagger_frac * period
            severed = _in_gap(xi_mid, period, params.L_fiber, shift)
            if not severed:
                elements.append(
                    {"etype": "tow", "ply": ply, "a": a, "b": b, "dx": tx, "dy": ty, "k": k_ax}
                )
            else:
                elements.append(
                    {"etype": "tow_severed", "ply": ply, "a": a, "b": b, "dx": tx, "dy": ty, "k": 0.0}
                )
            elements.append(
                {"etype": "m_tshear", "ply": ply, "a": a, "b": b, "dx": nx, "dy": ny, "k": k_tow_shear}
            )
        # Matrix coupling to the adjacent strip (shear along t, normal along n).
        c = index.get((j, k + 1))
        if c is not None:
            elements.append(
                {"etype": "m_shear", "ply": ply, "a": a, "b": c, "dx": tx, "dy": ty, "k": k_shear}
            )
            elements.append(
                {"etype": "m_normal", "ply": ply, "a": a, "b": c, "dx": nx, "dy": ny, "k": k_normal}
            )

    return nodes, index, elements


def build_mesh(params: Params2D) -> Mesh2D:
    all_nodes: list[tuple[float, float]] = []
    all_ply: list[int] = []
    elements: list[dict] = []

    for ply in range(params.n_plies):
        nodes, _index, elems = build_ply_mesh(params, ply, node_offset=len(all_nodes))
        all_nodes.extend(nodes)
        all_ply.extend([ply] * len(nodes))
        elements.extend(elems)

    mesh = Mesh2D(
        nodes=np.array(all_nodes, dtype=float).reshape(-1, 2),
        node_ply=np.array(all_ply, dtype=int),
        elements=elements,
    )
    return mesh
