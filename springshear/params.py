from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LayupName = Literal[
    "[0/90]s",
    "[0/45/90]s",
    "[0/45/90]",
    "quasi_isotropic",
]


@dataclass
class Params:
    L_fiber: float = 50e-3
    n_fibers: int = 4
    dx: float = 0.1e-3
    n_rows: int = 5
    row_width: float = 4e-3
    resin_gap: float = 0.5e-3
    L: float = field(init=False)
    eps0: float = 0.01
    E_tow: float = 140e9
    D_f: float = 4e-3
    t_f: float = 0.1e-3
    A_tow: float = field(init=False)
    G_m: float = 1.0e9
    E_m: float = 3.0e9
    t_eff: float = 0.10e-3
    b_eff: float = 1.0e-3
    row_stagger_x: float = field(init=False)
    n_plies: int = 1
    ply_angles: list[float] = field(default_factory=list)
    ply_gap_shifts: list[float] = field(default_factory=list)
    ply_offset_x: float = field(init=False)
    ply_offset_y: float = field(init=False)
    bc_mode: Literal["fixed", "periodic"] = "fixed"
    periodic_x: bool = True
    periodic_y: bool = True
    periodic_z: bool = False

    def __post_init__(self) -> None:
        self.L = (self.L_fiber + self.resin_gap) * self.n_fibers
        self.A_tow = self.D_f * self.t_f
        self.row_stagger_x = 0.5 * self.L_fiber
        self.ply_offset_x = 0.5 * self.L_fiber
        self.ply_offset_y = 0.5 * self.row_width
        self._normalize_ply_lists()

    @property
    def row_pitch(self) -> float:
        return self.row_width

    @property
    def A_m(self) -> float:
        return self.b_eff * self.t_eff

    def apply_stagger_preset(self, mode: Literal["aligned", "half_ply"]) -> None:
        if mode == "aligned":
            self.ply_offset_x = 0.0
            self.ply_offset_y = 0.0
        elif mode == "half_ply":
            self.ply_offset_x = 0.5 * self.L_fiber
            self.ply_offset_y = 0.5 * self.row_width

    def _normalize_ply_lists(self) -> None:
        if not self.ply_angles:
            self.ply_angles = [0.0] * self.n_plies
        elif len(self.ply_angles) < self.n_plies:
            pad = [self.ply_angles[-1]] * (self.n_plies - len(self.ply_angles))
            self.ply_angles = self.ply_angles + pad
        else:
            self.ply_angles = self.ply_angles[: self.n_plies]

        if not self.ply_gap_shifts:
            self.ply_gap_shifts = [0.0] * self.n_plies
        elif len(self.ply_gap_shifts) < self.n_plies:
            pad = [self.ply_gap_shifts[-1]] * (self.n_plies - len(self.ply_gap_shifts))
            self.ply_gap_shifts = self.ply_gap_shifts + pad
        else:
            self.ply_gap_shifts = self.ply_gap_shifts[: self.n_plies]

    def ply_angle(self, ply: int) -> float:
        return self.ply_angles[ply]

    def apply_layup(self, layup: LayupName | str) -> None:
        """Set ply count and orientations for common quasi-isotropic stacks."""
        if layup in ("[0/90]s", "0_90_s"):
            self.n_plies = 4
            self.ply_angles = [0.0, 90.0, 90.0, 0.0]
        elif layup in ("[0/45/90]s", "quasi_isotropic", "0_45_90_s"):
            self.n_plies = 4
            self.ply_angles = [0.0, 45.0, -45.0, 90.0]
        elif layup in ("[0/45/90]", "0_45_90"):
            self.n_plies = 3
            self.ply_angles = [0.0, 45.0, 90.0]
        else:
            raise ValueError(f"Unknown layup: {layup!r}")
        self.ply_gap_shifts = [0.0] * self.n_plies
        self._normalize_ply_lists()

    def set_ply_gap_shifts(self, shifts: list[float]) -> None:
        """Assign per-ply gap-pattern shifts as fractions of one brick period."""
        self.ply_gap_shifts = list(shifts)
        self._normalize_ply_lists()
