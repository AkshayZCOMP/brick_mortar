"""Sanity and regression checks for the 2D off-axis spring-network solver."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from springshear.spring2d.metrics2d import evaluate2d, tow_axial_stresses
from springshear.spring2d.params2d import Params2D, _fold_fiber_angle
from springshear.spring2d.solve2d import solve2d


def test_fiber_angle_fold():
    assert _fold_fiber_angle(0.0) == 0.0
    assert _fold_fiber_angle(45.0) == 45.0
    assert _fold_fiber_angle(90.0) == -90.0
    assert _fold_fiber_angle(135.0) == -45.0
    assert _fold_fiber_angle(-45.0) == -45.0


def test_no_singular_warning_all_angles():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for ang in (0.0, 30.0, 45.0, 60.0, 90.0):
            p = Params2D(ply_angles=[ang], load_dir="x")
            mesh, u = solve2d(p)
            assert np.isfinite(u).all(), f"angle {ang} produced non-finite u"


def test_uniform_strain_no_gap():
    """A continuous 0deg ply (no gaps) carries uniform tow stress under x-tension."""
    p = Params2D(ply_angles=[0.0], L_fiber=10.0, load_dir="x")  # L_fiber >> Lx => no gaps
    mesh, u = solve2d(p)
    sig = tow_axial_stresses(p, mesh, u)
    assert sig.size > 0
    cv = float(np.std(sig) / np.mean(sig))
    assert cv < 1e-6, f"continuous ply should be uniform, CV={cv}"
    assert np.mean(sig) > 0.0


def test_gaps_increase_nonuniformity():
    no_gap = Params2D(ply_angles=[0.0], L_fiber=10.0, load_dir="x")
    with_gap = Params2D(ply_angles=[0.0], load_dir="x")
    m0, u0 = solve2d(no_gap)
    m1, u1 = solve2d(with_gap)
    cv0 = evaluate2d(no_gap, m0, u0)["stress_cv"]
    cv1 = evaluate2d(with_gap, m1, u1)["stress_cv"]
    assert cv1 > cv0, f"severed gaps should raise stress CV ({cv1} vs {cv0})"


def test_transverse_ply_carries_little_axial():
    """A 90deg ply under x-tension carries far less axial tow stress than a 0deg ply."""
    p0 = Params2D(ply_angles=[0.0], L_fiber=10.0, load_dir="x")
    p90 = Params2D(ply_angles=[90.0], L_fiber=10.0, load_dir="x")
    m0, u0 = solve2d(p0)
    m9, u9 = solve2d(p90)
    s0 = tow_axial_stresses(p0, m0, u0, load_direction_only=False)
    s9 = tow_axial_stresses(p90, m9, u9, load_direction_only=False)
    assert abs(np.mean(s9)) < 0.2 * abs(np.mean(s0))


def test_stack_solves_and_interlaminar():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = Params2D(ply_angles=[0.0, 45.0, -45.0, 90.0], load_dir="x")
        mesh, u = solve2d(p)
        assert np.isfinite(u).all()
        il = sum(1 for e in mesh.elements if e["etype"] == "il_shear")
        assert il > 0, "multi-ply stack must have interlaminar coupling"
        metrics = evaluate2d(p, mesh, u)
        assert metrics["physically_sane"]


def test_load_direction_x_vs_y():
    """x-tension loads the 0deg ply; y-tension loads the 90deg ply."""
    angles = [0.0, 90.0, 90.0, 0.0]
    px = Params2D(ply_angles=angles, load_dir="x")
    py = Params2D(ply_angles=angles, load_dir="y")
    mx, ux = solve2d(px)
    my, uy = solve2d(py)
    cx = evaluate2d(px, mx, ux)["stress_cv"]
    cy = evaluate2d(py, my, uy)["stress_cv"]
    # [0/90]s is symmetric: the two modes should give comparable load-dir CV.
    assert np.isfinite(cx) and np.isfinite(cy)
    assert abs(cx - cy) / cx < 0.25, (cx, cy)


def run_all():
    tests = [
        test_fiber_angle_fold,
        test_no_singular_warning_all_angles,
        test_uniform_strain_no_gap,
        test_gaps_increase_nonuniformity,
        test_transverse_ply_carries_little_axial,
        test_stack_solves_and_interlaminar,
        test_load_direction_x_vs_y,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
