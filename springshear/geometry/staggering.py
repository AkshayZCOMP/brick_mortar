import numpy as np

from springshear.params import Params


def gap_intervals(L: float, Lf: float, g: float, shift: float) -> list[tuple[float, float]]:
    pitch = Lf + g
    k_max = int(np.ceil(L / pitch)) + 2
    gaps: list[tuple[float, float]] = []

    for k in range(k_max):
        x0 = (k + 1) * Lf + k * g - shift
        x1 = x0 + g
        a2 = max(0.0, x0)
        b2 = min(L, x1)
        if b2 > a2:
            gaps.append((round(a2, 6), round(b2, 6)))

    return gaps


def _normalize_angle(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def is_axial_ply(angle: float) -> bool:
    norm = _normalize_angle(angle)
    return abs(norm) < 1e-6 or abs(abs(norm) - 180.0) < 1e-6


def is_transverse_ply(angle: float) -> bool:
    norm = _normalize_angle(angle)
    return abs(abs(norm) - 90.0) < 1e-6


def row_gaps(params: Params, row: int, x_shift: float = 0.0) -> list[tuple[float, float]]:
    if row % 2 == 0:
        shift = x_shift
    else:
        shift = x_shift + params.row_stagger_x
    return gap_intervals(params.L, params.L_fiber, params.resin_gap, shift)


def ply_gap_shift(params: Params, ply: int) -> float:
    period = params.L_fiber + params.resin_gap
    frac = params.ply_gap_shifts[ply] if ply < len(params.ply_gap_shifts) else 0.0
    return ply_x_shift(params, ply) + frac * period


def row_gaps_for_ply(params: Params, ply: int, row: int) -> list[tuple[float, float]]:
    return row_gaps(params, row, ply_gap_shift(params, ply))


def boundary_gaps_for_ply(params: Params, ply: int, boundary: int) -> list[tuple[float, float]]:
    shift = ply_gap_shift(params, ply)
    if boundary % 2 == 1:
        shift += params.row_stagger_x
    return gap_intervals(params.L, params.L_fiber, params.resin_gap, shift)


def wrapped_ply_x_shift(params: Params, ply: int) -> float:
    period = params.L_fiber + params.resin_gap
    if period <= 0:
        return ply * params.ply_offset_x
    return (ply * params.ply_offset_x) % period


def wrapped_ply_y_shift(params: Params, ply: int) -> float:
    if params.periodic_y:
        period = params.n_rows * params.row_pitch
        if period <= 0:
            return ply * params.ply_offset_y
        return (ply * params.ply_offset_y) % period
    return ply * params.ply_offset_y


def ply_x_shift(params: Params, ply: int) -> float:
    return wrapped_ply_x_shift(params, ply)


def ply_y_shift(params: Params, ply: int) -> float:
    return wrapped_ply_y_shift(params, ply)


def row_y_center(params: Params, ply: int, row: int) -> float:
    return ply_y_shift(params, ply) + row * params.row_pitch + 0.5 * params.row_width


def fiber_coord(params: Params, ply: int, row: int, xmid: float) -> float:
    theta = np.deg2rad(params.ply_angle(ply))
    yc = row_y_center(params, ply, row)
    return xmid * np.cos(theta) + yc * np.sin(theta)


def is_in_gap_boundary(params: Params, ply: int, boundary: int, xmid: float) -> bool:
    gaps = boundary_gaps_for_ply(params, ply, boundary)
    return segment_in_gaps(xmid, gaps)


def is_in_gap(params: Params, ply: int, row: int, xmid: float) -> bool:
    angle = params.ply_angle(ply)
    if is_transverse_ply(angle):
        return False
    if is_axial_ply(angle):
        return segment_in_gaps(xmid, row_gaps_for_ply(params, ply, row))
    s = fiber_coord(params, ply, row, xmid)
    return segment_in_gaps(s, row_gaps_for_ply(params, ply, row))


def bridge_strength(params: Params) -> float:
    if params.E_tow <= 0 or params.A_tow <= 0:
        return 0.0
    return min(1.0, params.E_m / params.E_tow * params.A_m / params.A_tow)


def boundary_fiber_strength(params: Params, ply: int, boundary: int, xmid: float) -> float:
    if is_in_gap_boundary(params, ply, boundary, xmid):
        return bridge_strength(params)
    return 1.0


def fiber_strength(params: Params, ply: int, row: int, xmid: float) -> float:
    angle = params.ply_angle(ply)
    if is_transverse_ply(angle):
        vals: list[float] = []
        if row > 0:
            vals.append(boundary_fiber_strength(params, ply, row - 1, xmid))
        elif params.periodic_y and params.n_rows > 1:
            vals.append(boundary_fiber_strength(params, ply, params.n_rows - 1, xmid))
        if row < params.n_rows - 1:
            vals.append(boundary_fiber_strength(params, ply, row, xmid))
        elif params.periodic_y and params.n_rows > 1:
            vals.append(boundary_fiber_strength(params, ply, params.n_rows - 1, xmid))
        return max(vals) if vals else 1.0
    if is_in_gap(params, ply, row, xmid):
        return bridge_strength(params)
    return 1.0


def inter_ply_x_weight(
    params: Params,
    ply: int,
    row: int,
    xmid: float,
    partner_ply: int,
    partner_rows: list[int],
) -> float:
    """Coupling strength from one tow row to inter-ply matrix at x."""
    f_self = fiber_strength(params, ply, row, xmid)
    if not partner_rows:
        return f_self
    f_partner = max(fiber_strength(params, partner_ply, r, xmid) for r in partner_rows)
    return max(f_self, f_partner)


def segment_xmid(x: np.ndarray, node: int) -> float:
    if node < len(x) - 1:
        return 0.5 * (x[node] + x[node + 1])
    return 0.5 * (x[node - 1] + x[node])


def segment_in_gaps(xmid: float, gaps: list[tuple[float, float]]) -> bool:
    return any(x0 <= xmid <= x1 for x0, x1 in gaps)


def row_overlap(
    row_a: int,
    y_shift_a: float,
    row_b: int,
    y_shift_b: float,
    row_width: float,
    row_pitch: float,
) -> float:
    """Fraction of row_a band overlapping row_b band (0..1)."""
    a0 = y_shift_a + row_a * row_pitch
    a1 = a0 + row_width
    b0 = y_shift_b + row_b * row_pitch
    b1 = b0 + row_width
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    return overlap / row_width if row_width > 0 else 0.0
