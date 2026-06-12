"""Layup presets and parameter builders for quasi-isotropic gap-placement studies."""

from __future__ import annotations

import copy

from springshear.params import Params

# In-plane tension load cases. "0deg" applies tension along x (the as-defined
# layup); "90deg" applies tension along the orthogonal in-plane axis, realized
# by rotating every ply +90 deg and reusing the same x-tension solve.
LOAD_MODES = ("0deg", "90deg")


def _rotate_fiber_angle(angle: float, by_deg: float) -> float:
    """Rotate a fiber-line angle and fold to (-90, 90] (fibers are bidirectional)."""
    a = (angle + by_deg) % 180.0
    if a > 90.0:
        a -= 180.0
    return a


def load_mode_params(params: Params, mode: str) -> Params:
    """Return params for the requested in-plane tension mode.

    "0deg" returns params unchanged (tension along x). "90deg" returns a copy
    with every ply rotated +90 deg, which is equivalent to applying tension
    along the transverse in-plane axis. Per-ply gap patterns are preserved.
    """
    if mode in ("0deg", "0", "x"):
        return params
    if mode in ("90deg", "90", "y"):
        rotated = copy.deepcopy(params)
        rotated.ply_angles = [_rotate_fiber_angle(a, 90.0) for a in params.ply_angles]
        return rotated
    raise ValueError(f"Unknown load mode: {mode!r} (expected one of {LOAD_MODES})")


def base_quasi_params(
    layup: str,
    *,
    n_fibers: int = 2,
    n_rows: int = 3,
    dx: float = 0.2e-3,
    bc_mode: str = "fixed",
) -> Params:
    params = Params(
        n_fibers=n_fibers,
        n_rows=n_rows,
        dx=dx,
        bc_mode=bc_mode,
        periodic_x=bc_mode == "periodic",
        periodic_y=False,
        periodic_z=False,
    )
    params.apply_layup(layup)
    params.apply_stagger_preset("aligned")
    return params


def angle_groups(params: Params) -> dict[str, list[int]]:
    """Group ply indices by orientation for independent gap-shift sweeps."""
    groups: dict[str, list[int]] = {}
    for ply, angle in enumerate(params.ply_angles):
        key = f"{int(angle):+d}"
        groups.setdefault(key, []).append(ply)
    return groups


def set_group_gap_shifts(params: Params, shifts_by_angle: dict[str, float]) -> None:
    """Apply one gap-shift fraction to every ply sharing an orientation."""
    shifts = list(params.ply_gap_shifts)
    for angle_key, frac in shifts_by_angle.items():
        target = float(angle_key)
        for ply, angle in enumerate(params.ply_angles):
            if int(angle) == int(target):
                shifts[ply] = frac
    params.set_ply_gap_shifts(shifts)


def set_ply_stagger(params: Params, offset_x_frac: float, offset_y_frac: float) -> None:
    params.ply_offset_x = offset_x_frac * params.L_fiber
    params.ply_offset_y = offset_y_frac * params.row_pitch
