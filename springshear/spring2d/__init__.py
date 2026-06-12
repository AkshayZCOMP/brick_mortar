"""2D (u_x, u_y) spring-network extension of the spring-shear method.

A lightweight planar generalization of the 1D shear-lag model used to study how
resin-gap placement affects off-axis plies, where load transfer around severed
tows is genuinely two-dimensional. Tows are continuous strips at the ply angle
theta, resin gaps fully sever the tow (load detours through matrix shear), and
the matrix is represented by shear and transverse springs between adjacent
strips. Multiple plies are stacked and coupled by interlaminar in-plane shear.

The model is intentionally light (discrete spring network, not continuum FE) so
candidate geometries can be screened quickly before detailed FEA.
"""

from springshear.spring2d.mesh2d import build_mesh
from springshear.spring2d.metrics2d import evaluate2d
from springshear.spring2d.params2d import Params2D
from springshear.spring2d.solve2d import solve2d

__all__ = ["Params2D", "solve2d", "build_mesh", "evaluate2d"]
