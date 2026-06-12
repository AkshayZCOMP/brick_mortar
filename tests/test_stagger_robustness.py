"""Regression checks for 3D stagger robustness."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from springshear.geometry.staggering import (
    inter_ply_x_weight,
    is_in_gap,
    row_gaps,
    row_overlap,
    segment_xmid,
)
from springshear.metrics.objectives import evaluate_objective
from springshear.params import Params
from springshear.solve.run import solve


def _base_3d_params(bc_mode: str = "fixed") -> Params:
    return Params(
        n_plies=2,
        n_fibers=2,
        n_rows=3,
        dx=0.2e-3,
        bc_mode=bc_mode,
        periodic_x=bc_mode == "periodic",
        periodic_y=False,
        periodic_z=False,
    )


def test_row_overlap_sums_to_one():
    params = _base_3d_params()
    params.apply_stagger_preset("half_ply")
    y_shift = params.ply_offset_y
    for row in range(params.n_rows):
        raw = sum(
            row_overlap(row, y_shift, ipm_row, 0.0, params.row_width, params.row_pitch)
            for ipm_row in range(params.n_rows)
        )
        total = sum(
            row_overlap(row, y_shift, ipm_row, 0.0, params.row_width, params.row_pitch) / raw
            for ipm_row in range(params.n_rows)
            if raw > 0
            and row_overlap(row, y_shift, ipm_row, 0.0, params.row_width, params.row_pitch) > 0
        )
        assert abs(total - 1.0) < 1e-9, f"row {row} normalized y-overlap sum = {total}"


def test_inter_ply_x_weight():
    params = _base_3d_params()
    params.apply_stagger_preset("aligned")
    gaps = row_gaps(params, row=0, x_shift=0.0)
    xmid_gap = 0.5 * (gaps[0][0] + gaps[0][1])
    assert is_in_gap(params, 0, 0, xmid_gap)
    assert is_in_gap(params, 1, 0, xmid_gap)
    w_stacked = inter_ply_x_weight(params, 0, 0, xmid_gap, 1, [0])
    assert w_stacked < 0.1, f"stacked gap coupling should be weak, got {w_stacked}"

    params.apply_stagger_preset("half_ply")
    assert not is_in_gap(params, 1, 0, xmid_gap)
    w_half = inter_ply_x_weight(params, 0, 0, xmid_gap, 1, [0])
    assert w_half == 1.0


def test_segment_xmid_endpoints():
    import numpy as np

    x = np.array([0.0, 0.5, 1.0])
    assert segment_xmid(x, 0) == 0.25
    assert segment_xmid(x, 1) == 0.75
    assert segment_xmid(x, 2) == 0.75


def test_3d_aligned_vs_half():
    aligned_params = _base_3d_params(bc_mode="fixed")
    aligned_params.apply_stagger_preset("aligned")
    x, u, elems, _ = solve(aligned_params)
    m_aligned = evaluate_objective(aligned_params, x, u, elems)

    half_params = _base_3d_params(bc_mode="fixed")
    half_params.apply_stagger_preset("half_ply")
    x, u, elems, _ = solve(half_params)
    m_half = evaluate_objective(half_params, x, u, elems)

    assert m_aligned["physically_sane"], "aligned case should be physically sane"
    assert m_half["physically_sane"], "half_ply case should be physically sane"
    assert m_half["stress_cv"] < m_aligned["stress_cv"], "half_ply should lower stress CV"


def test_3d_periodic_physically_sane():
    for mode in ("aligned", "half_ply"):
        params = _base_3d_params(bc_mode="periodic")
        params.apply_stagger_preset(mode)
        x, u, elems, _ = solve(params)
        metrics = evaluate_objective(params, x, u, elems)
        assert metrics["physically_sane"], f"{mode} periodic case should be physically sane"


def test_2d_pbc_regression():
    params = Params(
        n_plies=1,
        bc_mode="periodic",
        periodic_x=True,
        periodic_y=False,
        periodic_z=False,
    )
    x, u, elems, _ = solve(params)
    metrics = evaluate_objective(params, x, u, elems)
    assert metrics["mean_tow_stress"] > 0.0
    assert metrics["physically_sane"]


def run_all():
    tests = [
        test_row_overlap_sums_to_one,
        test_inter_ply_x_weight,
        test_segment_xmid_endpoints,
        test_3d_aligned_vs_half,
        test_3d_periodic_physically_sane,
        test_2d_pbc_regression,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
