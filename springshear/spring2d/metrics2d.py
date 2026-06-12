"""Stress recovery and objectives for the 2D spring-network solver."""

from __future__ import annotations

import numpy as np

from springshear.spring2d.mesh2d import Mesh2D
from springshear.spring2d.params2d import Params2D


def is_load_aligned(params: Params2D, ply: int, tol: float = 1e-6) -> bool:
    """True if the ply's tows run parallel to the applied tension axis."""
    ang = params.ply_angle(ply)
    if params.load_dir == "x":
        return abs(ang) < tol
    return abs(abs(ang) - 90.0) < tol


def _load_axis(params: Params2D) -> tuple[int, float]:
    if params.load_dir == "x":
        return 0, params.Lx
    return 1, params.Ly


def tow_axial_stresses(
    params: Params2D,
    mesh: Mesh2D,
    u: np.ndarray,
    load_direction_only: bool = True,
    grip_margin: float | None = None,
) -> np.ndarray:
    """Tow axial stress (Pa) on intact tow segments, excluding the grip bands."""
    ux = u[0::2]
    uy = u[1::2]
    axis, span = _load_axis(params)
    if grip_margin is None:
        grip_margin = 2.0 * params.pitch

    stresses: list[float] = []
    for e in mesh.elements:
        if e["etype"] != "tow":
            continue
        if load_direction_only and not is_load_aligned(params, e["ply"]):
            continue
        mid = 0.5 * (mesh.nodes[e["a"], axis] + mesh.nodes[e["b"], axis])
        if mid < grip_margin or mid > span - grip_margin:
            continue
        strain = ((ux[e["b"]] - ux[e["a"]]) * e["dx"] + (uy[e["b"]] - uy[e["a"]]) * e["dy"]) / params.dxi
        stresses.append(params.E_tow * strain)
    return np.array(stresses, dtype=float)


def matrix_shear_stresses(params: Params2D, mesh: Mesh2D, u: np.ndarray) -> np.ndarray:
    """In-plane matrix shear stress (Pa) from inter-strip and along-tow shear springs."""
    ux = u[0::2]
    uy = u[1::2]
    h_m = params.matrix_gap_thickness
    taus: list[float] = []
    for e in mesh.elements:
        if e["etype"] not in ("m_shear", "m_tshear"):
            continue
        rel = (ux[e["b"]] - ux[e["a"]]) * e["dx"] + (uy[e["b"]] - uy[e["a"]]) * e["dy"]
        lever = h_m if e["etype"] == "m_shear" else params.dxi
        taus.append(params.G_m * rel / lever)
    return np.array(taus, dtype=float)


def effective_modulus(params: Params2D, mesh: Mesh2D, u: np.ndarray) -> float:
    """Apparent laminate modulus from the grip reaction: E = R / (eps0 * A).

    Computed from the assembled internal force f = K u summed over the loaded
    (far) face. As the RVE grows the grip boundary layer becomes a smaller
    fraction of the specimen, so this apparent modulus converges to the bulk
    value -- a clean scalar convergence indicator.
    """
    from springshear.spring2d.assemble2d import assemble
    from springshear.spring2d.bc2d import build_tension_bcs

    K = assemble(params, mesh)
    f = K @ u
    prescribed = build_tension_bcs(params, mesh)
    loaded = [d for d, v in prescribed.items() if v != 0.0]
    reaction = float(np.sum(f[loaded]))

    if params.load_dir == "x":
        width = params.Ly
    else:
        width = params.Lx
    area = width * params.n_plies * params.t_ply
    return reaction / (params.eps0 * area)


def stress_cv(stresses: np.ndarray) -> float:
    if stresses.size == 0:
        return float("nan")
    mu = float(np.mean(stresses))
    if abs(mu) < 1e-12:
        return float("nan")
    return float(np.std(stresses) / abs(mu))


def evaluate2d(params: Params2D, mesh: Mesh2D, u: np.ndarray) -> dict:
    tow_sig = tow_axial_stresses(params, mesh, u, load_direction_only=True)
    tau = matrix_shear_stresses(params, mesh, u)
    mean_sig = float(np.mean(tow_sig)) if tow_sig.size else 0.0
    tau_max = float(np.max(np.abs(tau))) if tau.size else 0.0
    finite = bool(np.isfinite(u).all())
    return {
        "stress_cv": stress_cv(tow_sig),
        "mean_tow_stress": mean_sig,
        "max_tow_stress": float(np.max(tow_sig)) if tow_sig.size else 0.0,
        "min_tow_stress": float(np.min(tow_sig)) if tow_sig.size else 0.0,
        "tau_matrix_max": tau_max,
        "n_load_tows": int(tow_sig.size),
        "load_dir": params.load_dir,
        "physically_sane": finite and mean_sig > 0.0 and tau_max < 1e12,
    }
