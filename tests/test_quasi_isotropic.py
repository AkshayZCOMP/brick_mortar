"""Sanity checks for multi-orientation layups."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from springshear.geometry.staggering import is_in_gap, is_in_gap_boundary, is_transverse_ply
from springshear.metrics.objectives import evaluate_objective
from springshear.params import Params
from springshear.solve.run import solve


def _quasi_params(layup: str) -> Params:
    params = Params(
        n_fibers=2,
        n_rows=3,
        dx=0.2e-3,
        bc_mode="fixed",
        periodic_x=False,
        periodic_y=False,
        periodic_z=False,
    )
    params.apply_layup(layup)
    params.apply_stagger_preset("aligned")
    return params


def test_layup_presets():
    p = _quasi_params("[0/90]s")
    assert p.n_plies == 4
    assert p.ply_angles == [0.0, 90.0, 90.0, 0.0]
    p.apply_layup("quasi_isotropic")
    assert p.ply_angles == [0.0, 45.0, -45.0, 90.0]


def test_90_degree_gap_detection():
    from springshear.geometry.staggering import boundary_gaps_for_ply

    params = Params(n_plies=1, ply_angles=[90.0], n_fibers=2, n_rows=3)
    assert is_transverse_ply(params.ply_angle(0))
    gaps = boundary_gaps_for_ply(params, 0, 0)
    x_gap = 0.5 * (gaps[0][0] + gaps[0][1])
    assert is_in_gap_boundary(params, 0, 0, x_gap)
    assert not is_in_gap(params, 0, 0, 0.0)


def test_quasi_layups_solve():
    for layup in ("[0/90]s", "[0/45/90]s"):
        params = _quasi_params(layup)
        x, u, elems, _ = solve(params)
        metrics = evaluate_objective(params, x, u, elems)
        assert metrics["physically_sane"], f"{layup} should be physically sane"
        assert metrics["mean_tow_stress"] > 0.0


def test_objective_is_load_direction_only():
    """stress_cv / mean_tow_stress must be measured on 0 deg plies only."""
    params = _quasi_params("[0/90]s")
    x, u, elems, _ = solve(params)
    metrics = evaluate_objective(params, x, u, elems)

    by_angle = metrics["mean_stress_by_angle"]
    assert "+0" in by_angle and "+90" in by_angle
    # 0 deg plies carry far more axial load than 90 deg plies under x-tension.
    assert by_angle["+0"] > by_angle["+90"]
    # Headline mean must match the 0 deg group, not a pooled average.
    assert abs(metrics["mean_tow_stress"] - by_angle["+0"]) / by_angle["+0"] < 0.05


def test_load_mode_rotation_and_symmetry():
    """90 deg mode rotates plies +90; [0/90]s is ~symmetric between the two modes."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "quasi_isotropic"))
    from layups import base_quasi_params, load_mode_params

    params = base_quasi_params("[0/45/90]s", bc_mode="fixed")
    rotated = load_mode_params(params, "90deg")
    assert load_mode_params(params, "0deg") is params
    assert rotated is not params
    assert params.ply_angles == [0.0, 45.0, -45.0, 90.0]
    # +90 folded to (-90, 90]: 0->90, 45->-45, -45->45, 90->0
    assert rotated.ply_angles == [90.0, -45.0, 45.0, 0.0]

    p090 = base_quasi_params("[0/90]s", bc_mode="fixed")
    cvs = {}
    for mode in ("0deg", "90deg"):
        sp = load_mode_params(p090, mode)
        x, u, elems, _ = solve(sp)
        cvs[mode] = evaluate_objective(sp, x, u, elems)["stress_cv"]
    assert abs(cvs["0deg"] - cvs["90deg"]) / cvs["0deg"] < 0.05, cvs


def run_all():
    tests = [
        test_layup_presets,
        test_90_degree_gap_detection,
        test_quasi_layups_solve,
        test_objective_is_load_direction_only,
        test_load_mode_rotation_and_symmetry,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
