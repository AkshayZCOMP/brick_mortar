from __future__ import annotations

import numpy as np

from springshear.geometry.staggering import is_axial_ply
from springshear.params import Params


def _ply_is_load_direction(params: Params, ply: int) -> bool:
    """True for 0/180 deg plies (fibers aligned with the applied x-strain)."""
    return is_axial_ply(params.ply_angle(ply))


def extract_tow_stresses(
    params: Params,
    u: np.ndarray,
    elems: list,
    exclude_boundary: int = 3,
    n_segments: int | None = None,
    load_direction_only: bool = True,
) -> np.ndarray:
    """Return tow axial stresses (Pa), excluding segments near x boundaries.

    By default only load-direction (0 deg) plies are included, since the
    stress-uniformity objective for brick-and-mortar staggering is meaningful
    on the fibers aligned with the applied strain. For a unidirectional stack
    (all plies 0 deg) this is identical to including every ply.
    """
    if n_segments is None:
        n_segments = max((e["i"] for e in elems if e["etype"] == "tow_axial"), default=0) + 1

    stresses: list[float] = []
    for e in elems:
        if e["etype"] != "tow_axial":
            continue
        if load_direction_only and not _ply_is_load_direction(params, e.get("ply", 0)):
            continue
        if e["i"] < exclude_boundary or e["i"] >= n_segments - exclude_boundary:
            continue
        F = e["k"] * (u[e["b"]] - u[e["a"]])
        stresses.append(F / e["A"])
    return np.array(stresses, dtype=float)


def mean_tow_stress_by_orientation(
    params: Params, u: np.ndarray, elems: list, exclude_boundary: int = 3
) -> dict[str, float]:
    """Mean tow axial stress (Pa) grouped by ply orientation, reported separately."""
    n_segments = max((e["i"] for e in elems if e["etype"] == "tow_axial"), default=0) + 1
    groups: dict[str, list[float]] = {}
    for e in elems:
        if e["etype"] != "tow_axial":
            continue
        if e["i"] < exclude_boundary or e["i"] >= n_segments - exclude_boundary:
            continue
        angle = params.ply_angle(e.get("ply", 0))
        key = f"{int(round(angle)):+d}"
        F = e["k"] * (u[e["b"]] - u[e["a"]])
        groups.setdefault(key, []).append(F / e["A"])
    return {k: float(np.mean(v)) for k, v in sorted(groups.items()) if v}


def stress_cv(stresses: np.ndarray) -> float:
    if stresses.size == 0:
        return float("inf")
    mu = np.mean(stresses)
    if abs(mu) < 1e-12:
        return float("inf")
    return float(np.std(stresses) / abs(mu))


def stress_cv_abs(stresses: np.ndarray) -> float:
    if stresses.size == 0:
        return float("inf")
    mu = np.mean(np.abs(stresses))
    if mu < 1e-12:
        return float("inf")
    return float(np.std(stresses) / mu)


def fiber_volume_fraction(params: Params, elems: list) -> float:
    n_axial = sum(1 for e in elems if e["etype"] == "tow_axial")
    n_bridge = sum(1 for e in elems if e["etype"] == "tow_bridge")
    total = n_axial + n_bridge
    if total == 0:
        return 0.0
    return n_axial / total


def effective_modulus(params: Params, mean_stress: float) -> float:
    if params.eps0 == 0:
        return 0.0
    return mean_stress / params.eps0


def max_shear_traction(params: Params, u: np.ndarray, elems: list) -> float:
    taus: list[float] = []
    patch = params.b_eff * params.dx
    for e in elems:
        if e["etype"] == "tm_shear" and e["row"] == e["r_int"]:
            F = e["k"] * (u[e["a"]] - u[e["m"]])
            taus.append(abs(F / patch))
        elif e["etype"] == "ip_tm_shear":
            F = e["k"] * (u[e["a"]] - u[e["m"]])
            taus.append(abs(F / patch))
    return max(taus) if taus else 0.0


def is_physically_sane(mean_tow_stress: float, tau_max: float) -> bool:
    return mean_tow_stress > 0.0 and tau_max < 1e12


def evaluate_objective(params: Params, x: np.ndarray, u: np.ndarray, elems: list) -> dict:
    n_seg = len(x) - 1
    # Stress-uniformity objective is measured on load-direction (0 deg) plies.
    tow_sig = extract_tow_stresses(params, u, elems, n_segments=n_seg, load_direction_only=True)
    mean_stress = float(np.mean(tow_sig)) if tow_sig.size else 0.0
    tau_max = max_shear_traction(params, u, elems)
    vf = fiber_volume_fraction(params, elems)
    e_eff = effective_modulus(params, mean_stress)
    e_eff_vf = e_eff / vf if vf > 0 else 0.0
    by_orientation = mean_tow_stress_by_orientation(params, u, elems)
    return {
        "stress_cv": stress_cv(tow_sig),
        "stress_cv_abs": stress_cv_abs(tow_sig),
        "mean_tow_stress": mean_stress,
        "max_tow_stress": float(np.max(tow_sig)) if tow_sig.size else 0.0,
        "min_tow_stress": float(np.min(tow_sig)) if tow_sig.size else 0.0,
        "E_eff": e_eff,
        "E_eff_vf": e_eff_vf,
        "fiber_vf": vf,
        "tau_max": tau_max,
        "target_stress": params.E_tow * params.eps0,
        "physically_sane": is_physically_sane(mean_stress, tau_max),
        "mean_stress_by_angle": by_orientation,
    }
