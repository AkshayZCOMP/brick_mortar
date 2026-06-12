"""Fixed-grip uniaxial tension boundary conditions for the 2D solver."""

from __future__ import annotations

import numpy as np

from springshear.spring2d.mesh2d import Mesh2D
from springshear.spring2d.params2d import Params2D


def build_tension_bcs(params: Params2D, mesh: Mesh2D) -> dict[int, float]:
    """Prescribe a fixed-grip uniaxial strain in params.load_dir.

    The loaded faces are fully clamped (both u_x and u_y) so the grips do not
    rotate or neck; the far face is displaced by eps0 * span.
    """
    nodes = mesh.nodes
    prescribed: dict[int, float] = {}

    if params.load_dir == "x":
        axis, span = 0, params.Lx
    elif params.load_dir == "y":
        axis, span = 1, params.Ly
    else:
        raise ValueError(f"Unknown load_dir: {params.load_dir!r}")

    coord = nodes[:, axis]
    tol = 0.6 * params.pitch
    lo_face = np.where(coord <= coord.min() + tol)[0]
    hi_face = np.where(coord >= coord.max() - tol)[0]
    u_far = params.eps0 * span

    for nd in lo_face:
        prescribed[2 * nd] = 0.0
        prescribed[2 * nd + 1] = 0.0
    for nd in hi_face:
        prescribed[2 * nd + axis] = u_far
        prescribed[2 * nd + (1 - axis)] = 0.0

    return prescribed
