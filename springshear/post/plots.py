from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from springshear.geometry.staggering import ply_x_shift, ply_y_shift, row_gaps, segment_in_gaps
from springshear.params import Params


def _collect_ply_stresses(
    params: Params, x: np.ndarray, u: np.ndarray, elems: list, ply: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = len(x)
    x_mid = 0.5 * (x[:-1] + x[1:])
    sig_tow = np.full((params.n_rows, n_nodes - 1), np.nan)
    sig_bridge = np.full((params.n_rows, n_nodes - 1), np.nan)
    A_m = params.A_m

    for e in elems:
        if e.get("ply", 0) != ply:
            continue
        if e["etype"] == "tow_axial":
            F = e["k"] * (u[e["b"]] - u[e["a"]])
            sig_tow[e["row"], e["i"]] = F / e["A"]
        elif e["etype"] == "tow_bridge":
            F = e["k"] * (u[e["b"]] - u[e["a"]])
            sig_bridge[e["row"], e["i"]] = F / A_m

    n_intf = params.n_rows - 1 if not params.periodic_y else params.n_rows
    tau_tm = np.full((n_intf, n_nodes), np.nan)
    for e in elems:
        if e.get("ply", 0) != ply:
            continue
        if e["etype"] == "tm_shear" and e["row"] == e["r_int"]:
            F = e["k"] * (u[e["a"]] - u[e["m"]])
            tau_tm[e["r_int"], e["i"]] = F / (params.b_eff * params.dx)

    return x_mid, sig_tow, sig_bridge, tau_tm


def plot_stack_stresses(
    params: Params,
    x: np.ndarray,
    u: np.ndarray,
    elems: list,
    plies: list[int] | None = None,
    show: bool = True,
):
    """Plot tow/bridge stress and tow-matrix traction for one or more plies."""
    if plies is None:
        plies = list(range(params.n_plies))

    n_plies = len(plies)
    fig, axes = plt.subplots(n_plies, 2, figsize=(12, 4 * n_plies), squeeze=False)

    for idx, ply in enumerate(plies):
        x_mid, sig_tow, sig_bridge, tau_tm = _collect_ply_stresses(params, x, u, elems, ply)
        sig_row = np.zeros((params.n_rows, len(x_mid)))
        mt = ~np.isnan(sig_tow)
        mb = ~np.isnan(sig_bridge)
        sig_row[mt] = sig_tow[mt]
        sig_row[mb] = sig_bridge[mb]

        ax_tow, ax_tau = axes[idx]
        for row in range(params.n_rows):
            ax_tow.plot(x_mid, sig_tow[row] / 1e9, label=f"row {row}")
        ax_tow.set(title=f"Ply {ply}: tow axial stress (GPa)", xlabel="x")
        ax_tow.grid(True)
        ax_tow.legend(fontsize=7)

        for r in range(tau_tm.shape[0]):
            ax_tau.plot(x, tau_tm[r] / 1e6, label=f"intf {r}")
        ax_tau.plot(x_mid, np.mean(sig_row, axis=0) / 1e9, "k--", lw=1.5, label="avg row")
        ax_tau.set(title=f"Ply {ply}: traction / avg stress", xlabel="x")
        ax_tau.grid(True)
        ax_tau.legend(fontsize=7)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def build_stack_stress_bands(
    params: Params, x: np.ndarray, u: np.ndarray, elems: list
) -> list[dict]:
    """Collect per-band stress strips for FEA-style heatmaps (x along fiber, y transverse)."""
    bands: list[dict] = []
    x_edges = x
    x_mid = 0.5 * (x[:-1] + x[1:])

    for ply in range(params.n_plies):
        _, sig_tow, sig_bridge, _ = _collect_ply_stresses(params, x, u, elems, ply)
        for row in range(params.n_rows):
            y0 = ply_y_shift(params, ply) + row * params.row_pitch
            y1 = y0 + params.row_width
            stress = np.zeros(len(x_mid))
            for j in range(len(x_mid)):
                if not np.isnan(sig_tow[row, j]):
                    stress[j] = sig_tow[row, j]
                elif not np.isnan(sig_bridge[row, j]):
                    stress[j] = sig_bridge[row, j]
                else:
                    stress[j] = np.nan

            gaps = row_gaps(params, row, ply_x_shift(params, ply))
            in_gap = np.array([segment_in_gaps(xm, gaps) for xm in x_mid])

            bands.append(
                {
                    "ply": ply,
                    "row": row,
                    "y0": y0,
                    "y1": y1,
                    "stress": stress,
                    "in_gap": in_gap,
                }
            )
    return bands, x_edges


def _draw_band_mesh(ax, x_edges: np.ndarray, y0: float, y1: float, color: str = "0.35", lw: float = 0.4):
    """Draw FEA-style element edges on a single tow-row band."""
    segs = []
    for i in range(len(x_edges) - 1):
        x0, x1 = x_edges[i], x_edges[i + 1]
        segs.extend([[(x0, y0), (x1, y0)], [(x0, y1), (x1, y1)], [(x0, y0), (x0, y1)], [(x1, y0), (x1, y1)]])
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=3))


def plot_stress_heatmap(
    params: Params,
    x: np.ndarray,
    u: np.ndarray,
    elems: list,
    ax: plt.Axes | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    scale: float = 1e-9,
    unit: str = "GPa",
    cmap: str = "jet",
    show_gaps: bool = True,
    show: bool = True,
    out_path: str | None = None,
):
    """FEA-style axial stress heatmap: x = fiber direction, y = transverse stack position."""
    bands, x_edges = build_stack_stress_bands(params, x, u, elems)
    all_sig = np.concatenate([b["stress"] for b in bands if np.any(~np.isnan(b["stress"]))])
    if vmin is None:
        vmin = float(np.nanmin(all_sig)) * scale
    if vmax is None:
        vmax = float(np.nanmax(all_sig)) * scale
    norm = Normalize(vmin=vmin, vmax=vmax)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4 + 0.35 * params.n_plies * params.n_rows))
    else:
        fig = ax.figure

    pcm = None
    for band in bands:
        z = band["stress"] * scale
        z2d = z.reshape(1, -1)
        pcm = ax.pcolormesh(
            x_edges,
            [band["y0"], band["y1"]],
            z2d,
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="none",
            antialiased=False,
            zorder=1,
        )
        _draw_band_mesh(ax, x_edges, band["y0"], band["y1"])

        if show_gaps:
            for j, is_gap in enumerate(band["in_gap"]):
                if not is_gap:
                    continue
                ax.fill_between(
                    [x_edges[j], x_edges[j + 1]],
                    band["y0"],
                    band["y1"],
                    facecolor="none",
                    hatch="///",
                    edgecolor="0.45",
                    linewidth=0.2,
                    zorder=2,
                )

    y_vals = [b["y0"] for b in bands] + [b["y1"] for b in bands]
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(min(y_vals), max(y_vals))
    ax.set_xlabel("x (m) — fiber direction")
    ax.set_ylabel("y (m) — transverse")
    ax.set_title(title or "Axial stress (FEA view)")
    ax.set_aspect("auto")

    standalone = ax.get_subplotspec() is None
    if standalone and pcm is not None:
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label(f"Axial stress ({unit})")

    if out_path and standalone:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show and standalone:
        plt.show()
    return fig, ax, pcm


def plot_fixed_tension_heatmap_compare(
    aligned: dict,
    half_ply: dict,
    out_path: str | None = None,
    show: bool = True,
    cmap: str = "jet",
):
    """Side-by-side FEA-style stress heatmaps with shared color scale."""
    cases = [("aligned", aligned), ("half_ply", half_ply)]
    all_stress: list[np.ndarray] = []
    for _, case in cases:
        bands, _ = build_stack_stress_bands(case["params"], case["x"], case["u"], case["elems"])
        for band in bands:
            all_stress.append(band["stress"])

    combined = np.concatenate([s for s in all_stress if np.any(~np.isnan(s))])
    vmin = float(np.nanmin(combined)) * 1e-9
    vmax = float(np.nanmax(combined)) * 1e-9

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained", sharey=True)
    mappable = None
    for ax, (label, case) in zip(axes, cases):
        _, _, mappable = plot_stress_heatmap(
            case["params"],
            case["x"],
            case["u"],
            case["elems"],
            ax=ax,
            title=f"{label} — axial stress",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            show=False,
        )

    if mappable is not None:
        fig.colorbar(mappable, ax=axes, pad=0.02, fraction=0.035, label="Axial stress (GPa)")

    fig.suptitle(
        f"Fixed-end tension  |  $\\varepsilon_0$={aligned['params'].eps0:.3f}  |  "
        f"hatched = resin gap",
        fontsize=11,
    )
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _mean_row_stress(sig_tow: np.ndarray, sig_bridge: np.ndarray) -> np.ndarray:
    sig_row = np.zeros(sig_tow.shape)
    mt = ~np.isnan(sig_tow)
    mb = ~np.isnan(sig_bridge)
    sig_row[mt] = sig_tow[mt]
    sig_row[mb] = sig_bridge[mb]
    return np.nanmean(sig_row, axis=0)


def plot_fixed_tension_compare(
    aligned: dict,
    half_ply: dict,
    out_path: str | None = None,
    show: bool = True,
):
    """Side-by-side visualization of aligned vs half-ply stagger under fixed tension."""
    modes = [("aligned", aligned), ("half_ply", half_ply)]
    n_plies = aligned["params"].n_plies
    fig = plt.figure(figsize=(14, 3 + 3 * n_plies), layout="constrained")
    gs = fig.add_gridspec(1 + n_plies, 3, width_ratios=[1, 1, 0.85])

    ax_metrics = fig.add_subplot(gs[0, :])
    labels = ["aligned", "half_ply"]
    cv_vals = [aligned["metrics"]["stress_cv"], half_ply["metrics"]["stress_cv"]]
    tau_vals = [
        aligned["metrics"]["tau_max"] / 1e6,
        half_ply["metrics"]["tau_max"] / 1e6,
    ]
    x_pos = np.arange(len(labels))
    w = 0.35
    ax_metrics.bar(x_pos - w / 2, cv_vals, w, label="stress CV", color="steelblue")
    ax_metrics.bar(x_pos + w / 2, tau_vals, w, label=r"$\tau_{max}$ (MPa)", color="coral")
    ax_metrics.set_xticks(x_pos, labels)
    ax_metrics.set_title("Fixed-end tension: aligned vs half-ply stagger")
    ax_metrics.legend()
    ax_metrics.grid(True, axis="y", alpha=0.4)

    for ply in range(n_plies):
        for col, (label, case) in enumerate(modes):
            params, x, u, elems = case["params"], case["x"], case["u"], case["elems"]
            x_mid, sig_tow, sig_bridge, tau_tm = _collect_ply_stresses(params, x, u, elems, ply)
            ax = fig.add_subplot(gs[1 + ply, col])
            for row in range(params.n_rows):
                ax.plot(x_mid, sig_tow[row] / 1e9, label=f"row {row}")
            ax.plot(
                x_mid,
                _mean_row_stress(sig_tow, sig_bridge) / 1e9,
                "k--",
                lw=1.5,
                label="avg",
            )
            ax.set_title(f"{label} — ply {ply} tow stress (GPa)")
            ax.set_xlabel("x (m)")
            ax.grid(True, alpha=0.4)
            ax.legend(fontsize=7)

        ax_avg = fig.add_subplot(gs[1 + ply, 2])
        for label, case, color in [
            ("aligned", aligned, "tab:blue"),
            ("half_ply", half_ply, "tab:orange"),
        ]:
            params, x, u, elems = case["params"], case["x"], case["u"], case["elems"]
            x_mid, sig_tow, sig_bridge, _ = _collect_ply_stresses(params, x, u, elems, ply)
            ax_avg.plot(
                x_mid,
                _mean_row_stress(sig_tow, sig_bridge) / 1e9,
                color=color,
                lw=2,
                label=label,
            )
        ax_avg.set_title(f"Ply {ply} avg stress overlay")
        ax_avg.set_xlabel("x (m)")
        ax_avg.legend(fontsize=8)
        ax_avg.grid(True, alpha=0.4)

    fig.suptitle(
        f"eps0={aligned['params'].eps0:.3f}  |  "
        f"CV improvement: "
        f"{(cv_vals[0] - cv_vals[1]) / cv_vals[0] * 100:.1f}%",
        fontsize=10,
    )
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_ply_stresses(params: Params, x: np.ndarray, u: np.ndarray, elems: list, show: bool = True):
    x_mid, sig_tow, sig_bridge, tau_tm = _collect_ply_stresses(params, x, u, elems, ply=0)
    sig_row = np.zeros(sig_tow.shape)
    mt = ~np.isnan(sig_tow)
    mb = ~np.isnan(sig_bridge)
    sig_row[mt] = sig_tow[mt]
    sig_row[mb] = sig_bridge[mb]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for row in range(params.n_rows):
        axes[0, 0].plot(x_mid, sig_tow[row] / 1e9, label=f"row {row}")
    axes[0, 0].set(title="Tow axial stress (GPa)", xlabel="x")

    for row in range(params.n_rows):
        mask = ~np.isnan(sig_bridge[row])
        if np.any(mask):
            axes[0, 1].scatter(x_mid[mask], sig_bridge[row, mask] / 1e9, s=12, label=f"row {row}")
    axes[0, 1].set(title="Bridge stress (GPa)", xlabel="x")

    for r in range(tau_tm.shape[0]):
        axes[1, 0].plot(x, tau_tm[r] / 1e6, label=f"intf {r}")
    axes[1, 0].set(title="Tow-matrix traction (MPa)", xlabel="x")

    axes[1, 1].plot(x_mid, np.mean(sig_row, axis=0) / 1e9, "k", lw=2)
    axes[1, 1].set(title="Average row stress (GPa)", xlabel="x")

    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(fontsize=7)
        ax.grid(True)

    plt.tight_layout()
    if show:
        plt.show()
    return fig
