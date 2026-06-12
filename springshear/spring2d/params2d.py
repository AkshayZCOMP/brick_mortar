"""Parameters for the 2D spring-network off-axis solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


def _fold_fiber_angle(angle: float) -> float:
    """Fold a fiber-line angle to (-90, 90] (fibers are bidirectional).

    0 -> 0, 45 -> 45, 90 -> -90, 135 -> -45, -45 -> -45.
    """
    return ((angle + 90.0) % 180.0) - 90.0


@dataclass
class Params2D:
    # --- RVE footprint (shared by every ply, in-plane x-y) ---
    Lx: float = 20.0e-3
    Ly: float = 20.0e-3

    # --- Brick-and-mortar tow geometry (measured along the tow axis) ---
    L_fiber: float = 8.0e-3       # intact tow length between gaps
    resin_gap: float = 0.8e-3     # severed-gap length along the tow
    tow_width: float = 1.8e-3     # strip width (perpendicular to tow axis)
    pitch: float = 2.0e-3         # center-to-center strip spacing (perp.)
    dxi: float = 0.4e-3           # node spacing along the tow axis
    t_ply: float = 0.2e-3         # ply thickness (out-of-plane)

    # --- Stagger / gap placement ---
    strip_stagger_frac: float = 0.5   # brick stagger between adjacent strips (fraction of period)

    # --- Materials ---
    E_tow: float = 140.0e9
    G_m: float = 1.0e9
    E_m: float = 3.0e9

    # Dimensionless calibration multipliers tuned against continuum FEA so the
    # spring network reproduces the measured gap knockdown (shear_lag_factor, on
    # the matrix shear-lag springs that bridge a severed tow) and the transverse
    # modulus (transverse_factor, on the inter-strip normal springs). Both are
    # 1.0 from first principles; see examples/offaxis_2d/calibrate.py.
    shear_lag_factor: float = 1.0
    transverse_factor: float = 1.0

    # --- Laminate ---
    ply_angles: list[float] = field(default_factory=lambda: [0.0])
    # Per-ply gap placement: longitudinal shift of the gap pattern along the tow
    # axis (fraction of the brick period) ...
    ply_gap_shifts: list[float] = field(default_factory=list)
    # ... and transverse shift of the whole tow-strip lattice perpendicular to
    # the tows (fraction of the strip pitch). This offsets the rows of one layer
    # relative to another so tow boundaries/gaps need not stack through-thickness.
    ply_transverse_shifts: list[float] = field(default_factory=list)
    interlaminar_thickness: float = 0.05e-3

    # --- Loading ---
    eps0: float = 0.01
    load_dir: Literal["x", "y"] = "x"

    def __post_init__(self) -> None:
        self.ply_angles = [_fold_fiber_angle(a) for a in self.ply_angles]
        self._normalize_lists()

    def _normalize_lists(self) -> None:
        self.ply_gap_shifts = self._fit_list(self.ply_gap_shifts)
        self.ply_transverse_shifts = self._fit_list(self.ply_transverse_shifts)

    def _fit_list(self, values: list[float]) -> list[float]:
        n = self.n_plies
        if not values:
            return [0.0] * n
        if len(values) < n:
            return values + [values[-1]] * (n - len(values))
        return values[:n]

    @property
    def n_plies(self) -> int:
        return len(self.ply_angles)

    @property
    def period(self) -> float:
        return self.L_fiber + self.resin_gap

    @property
    def matrix_gap_thickness(self) -> float:
        """In-plane matrix ligament thickness between adjacent strips."""
        return max(self.pitch - self.tow_width, 0.05 * self.pitch)

    @property
    def A_tow(self) -> float:
        """Tow strip cross-section (width x ply thickness)."""
        return self.tow_width * self.t_ply

    def ply_angle(self, ply: int) -> float:
        return self.ply_angles[ply]

    def with_load_dir(self, load_dir: Literal["x", "y"]) -> "Params2D":
        import copy

        p = copy.deepcopy(self)
        p.load_dir = load_dir
        return p
