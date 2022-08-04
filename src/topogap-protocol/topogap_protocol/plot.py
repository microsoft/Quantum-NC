##
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MICROSOFT QUANTUM NON-COMMERCIAL License.
##
from typing import Optional, Tuple

import holoviews as hv
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import display
from ipywidgets import IntSlider, SelectionSlider, interaction, interactive

from topogap_protocol.specs import (
    CalibrationData,
    PhaseOneData,
    PhaseTwoRegionData,
    RFData,
    complex_signal_at,
)

# needed to make the interactive plots work in Jupyterlab
interaction.show_inline_matplotlib_plots()


def calibration_check(
    phase_one: PhaseOneData,
    calibration_data: CalibrationData,
    rf_data: RFData,
    force_recalculate=False,
) -> interactive:
    data = phase_one.calibration_check(calibration_data, rf_data, force_recalculate)

    def _plot_distance_from_calibration(name, side):
        fig, ax = plt.subplots()
        ax.plot(
            rf_data.f_values,
            data["distance"].loc[name][side].values,
            c="k",
            linestyle="-",
            linewidth=0.5,
            marker=".",
        )
        ax.set_xlabel("Field [T]")
        ax.set_ylabel("Avg distance [e^2/h]")
        plt.grid()
        plt.show()

    widget = interactive(
        _plot_distance_from_calibration,
        side=list(calibration_data.get_standard_xarray_names()),
        name=list(rf_data.get_standard_xarray_names()),
    )
    return widget


def compare_rf_data_and_calibration(
    rf_data: RFData, calibration_data: CalibrationData, nsamples: int = 500
) -> interactive:
    # Pick the random indices:
    dims = rf_data.data["lc_lb"].sel(f=0).dims  # TODO: make this not hard-coded
    lim = np.product(list(dims.values()))
    idxs = np.random.randint(lim, size=nsamples)

    def _plot_data_on_complex_plane(side, name, f_idx):
        f = rf_data.f_values[f_idx]
        # calibration data points
        cal = calibration_data.data[side].sel(f=f, method="nearest")
        zcal = (cal["re"] + 1j * cal["im"]).values

        # rf data point (a sample)
        rf = rf_data.data[name].sel(f=f, method="nearest")
        zrf_sample = complex_signal_at(side, rf).values.flatten()[idxs]

        # plot on complex plane
        fig, ax = plt.subplots()
        ax.scatter(zrf_sample.real, zrf_sample.imag, marker=".", c="k", label="rf")
        ax.scatter(zcal.real, zcal.imag, marker="o", c="r", label="calibration")
        ax.set_xlabel("Real part")
        ax.set_ylabel("Imaginary part")
        ax.set_title(f"{name}, {side}, $B={f}$ T")
        ax.legend()
        fig.show()

    widget = interactive(
        _plot_data_on_complex_plane,
        side=list(calibration_data.get_standard_xarray_names()),
        name=list(rf_data.get_standard_xarray_names()),
        f_idx=IntSlider(min=0, max=len(rf_data.f_values) - 1),
    )
    return widget


def joined_zbp_probability(
    zbp_ds,
    zbp_threshold,
    plunger_name: str = "p",
    field_name: str = "f",
    zbp_name: str = "zbp",
    figsize: Tuple[float, float] = (5.1, 3.15),
    title=None,
    fname=None,
    show=True,
) -> mpl.figure.Figure:
    _set_attrs(zbp_ds, plunger_name, field_name)
    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    cmap = plt.get_cmap("viridis", 2)
    im = zbp_ds[zbp_name].plot.imshow(
        x=field_name, y=plunger_name, ax=ax, add_colorbar=False, cmap=cmap
    )
    ax.set_title(title or "Joint probability")
    im.set_clim(0, 1)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(
        [
            fr"$P(\mathrm{{ZBP}})<{zbp_threshold:.1f}$",
            fr"$P(\mathrm{{ZBP}}) \geq {zbp_threshold:.1f}$",
        ]
    )
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _set_attrs(ds, plunger_name, field_name):
    ds.left.attrs["long_name"] = "Probability of ZBP"
    ds.right.attrs["long_name"] = "Probability of ZBP"

    ds[field_name].attrs["units"] = "T"
    ds[field_name].attrs["long_name"] = "$B$"
    ds[plunger_name].attrs["units"] = "V"
    ds[plunger_name].attrs["long_name"] = r"$V_\mathrm{gate}$"


def left_right_zbp_probability(
    zbp_ds,
    zbp_threshold,
    plunger_name: str = "p",
    field_name: str = "f",
    figsize: Tuple[float, float] = (5.1, 3.15),
    discrete=False,
    fname=None,
) -> mpl.figure.Figure:
    _set_attrs(zbp_ds, plunger_name, field_name)
    info = [
        ("left", "Left side"),
        ("right", "Right side"),
    ]

    fig, axs = plt.subplots(ncols=2, figsize=figsize, sharey=True)
    cmap = plt.get_cmap("viridis", 2 if discrete else None)
    for i, ax in enumerate(axs):
        key, title = info[i]
        label = "abcd"[i]
        ax.text(
            x=0.02,
            y=0.97,
            s=f"({label})",
            color="w",
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment="top",
            horizontalalignment="left",
        )

        im = zbp_ds[key].plot.imshow(
            x=field_name, y=plunger_name, ax=ax, add_colorbar=False, cmap=cmap
        )
        ax.set_title(title)
        im.set_clim(0, 1)

    if discrete:
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(
            [
                fr"$P(\mathrm{{ZBP}})<{zbp_threshold:.1f}$",
                fr"$P(\mathrm{{ZBP}}) \geq {zbp_threshold:.1f}$",
            ]
        )
    else:
        cbar = fig.colorbar(im, ax=axs[1])
        cbar.set_label("Probability of ZBP")
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()


def _set_attrs_phase_one(phase_one: PhaseOneData) -> None:
    left = phase_one.data["lc_lb"]
    right = phase_one.data["rc_rb"]
    left["g_ll"].attrs["units"] = "$e^2/h$"
    left["g_ll"].attrs["long_name"] = "$G_{ll}$"
    right["g_rr"].attrs["units"] = "$e^2/h$"
    right["g_rr"].attrs["long_name"] = "$G_{rr}$"
    for s in (left, right):
        s["f"].attrs["units"] = "T"
        s["f"].attrs["long_name"] = "$B$"
        s["p"].attrs["long_name"] = r"$V_\mathrm{gate}$"
        s["lb"].attrs["long_name"] = r"$I_\mathrm{b,l}$"
        s["rb"].attrs["long_name"] = r"$I_\mathrm{b,r}$"
        s["rb"].attrs["units"] = "eV"
        s["lb"].attrs["units"] = "eV"
        for k in ("rc", "lc", "p"):
            s[k].attrs["units"] = "V"


def phase_one_interactive(
    phase_one: PhaseOneData,
    norm: Optional[colors.Normalize] = None,
    b_vs_f: bool = True,
    p_vs_f: bool = True,
    p_vs_b: bool = True,
) -> interactive:
    _set_attrs_phase_one(phase_one)
    left = phase_one.data["lc_lb"]
    right = phase_one.data["rc_rb"]

    ps = left.coords["p"].values
    lcs = left.coords["lc"].values
    rcs = right.coords["rc"].values
    lbs = left.coords["lb"].values
    rbs = right.coords["rb"].values
    fs = left.coords["f"].values

    def plot(left_sel, right_sel):
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
        kwargs = dict(norm=norm, cmap="viridis", vmin=0)
        left_sel["g_ll"].plot.imshow(ax=axs[0], **kwargs)
        right_sel["g_rr"].plot.imshow(ax=axs[1], **kwargs)
        axs[0].set_title("Left side")
        axs[1].set_title("Right side")
        plt.tight_layout()
        plt.show()
        return fig

    def _plot_b_vs_f(p, lc, rc):
        left_sel = left.sel({"p": p, "lc": lc}).squeeze()
        right_sel = right.sel({"p": p, "rc": rc}).squeeze()
        return plot(left_sel, right_sel)

    def _plot_p_vs_f(lb, rb, lc, rc):
        left_sel = left.sel({"lb": lb, "lc": lc}).squeeze()
        right_sel = right.sel({"rb": rb, "rc": rc}).squeeze()
        return plot(left_sel, right_sel)

    def _plot_p_vs_b(f, lc, rc):
        left_sel = left.sel({"f": f, "lc": lc}).squeeze()
        right_sel = right.sel({"f": f, "rc": rc}).squeeze()
        return plot(left_sel, right_sel)

    if b_vs_f:
        display(
            interactive(
                _plot_b_vs_f,
                p=SelectionSlider(options=list(ps)),
                lc=SelectionSlider(options=list(lcs)),
                rc=SelectionSlider(options=list(rcs)),
            )
        )

    if p_vs_f:
        display(
            interactive(
                _plot_p_vs_f,
                lb=SelectionSlider(options=list(lbs)),
                rb=SelectionSlider(options=list(rbs)),
                lc=SelectionSlider(options=list(lcs)),
                rc=SelectionSlider(options=list(rcs)),
            )
        )

    if p_vs_b:
        display(
            interactive(
                _plot_p_vs_b,
                f=SelectionSlider(options=list(fs)),
                lc=SelectionSlider(options=list(lcs)),
                rc=SelectionSlider(options=list(rcs)),
            )
        )


def phase_two_interactive(
    region: PhaseTwoRegionData, norm: Optional[colors.Normalize] = None, unit=None
) -> interactive:
    left = region.data["left"]
    right = region.data["right"]
    ps = left.coords["p"].values
    lcs = left.coords["lc"].values
    rcs = right.coords["rc"].values
    label = unit or ""

    def _plot(p, lc, rc):
        left_sel = left.sel({"p": p, "lc": lc}).squeeze()
        right_sel = right.sel({"p": p, "rc": rc}).squeeze()
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 5))
        left_sel["g_ll"].squeeze().transpose("lb", "f").plot.imshow(
            ax=axs[0, 0],
            add_colorbar=True,
            cbar_kwargs=dict(label=label),
            cmap="viridis",
            vmin=0,
        )
        axs[0, 0].set_title("$G_{ll}$")
        right_sel["g_rr"].squeeze().transpose("rb", "f").plot.imshow(
            ax=axs[0, 1],
            add_colorbar=True,
            cbar_kwargs=dict(label=label),
            cmap="viridis",
            vmin=0,
        )
        axs[0, 1].set_title("$G_{rr}$")

        left_sel["g_rl"].squeeze().transpose("lb", "f").plot.imshow(
            ax=axs[1, 0], norm=norm, add_colorbar=True, cbar_kwargs=dict(label=label)
        )
        axs[1, 0].set_title("$G_{rl}$")

        right_sel["g_lr"].squeeze().transpose("rb", "f").plot.imshow(
            ax=axs[1, 1], norm=norm, add_colorbar=True, cbar_kwargs=dict(label=label)
        )
        axs[1, 1].set_title("$G_{lr}$")

        plt.tight_layout()
        plt.show()
        return fig

    return interactive(
        _plot,
        p=SelectionSlider(options=list(ps)),
        lc=SelectionSlider(options=list(lcs)),
        rc=SelectionSlider(options=list(rcs)),
    )


def zbps(
    zbp_ds,
    ncols: int = 2,
    zbp_name: str = "zbp",
    plunger_name: str = "p",
    field_name: str = "f",
) -> hv.Layout:
    def _plot(which):
        hv_ds = hv.Dataset(zbp_ds[which])
        label = f"ZBP_{which}" if which != zbp_name else "ZBP_combined"
        return hv_ds.to(
            hv.Image, kdims=[field_name, plunger_name], vdims=[which], label=label
        ).options(aspect="square", cmap="coolwarm", tools=["hover"])

    return (_plot("right") + _plot("left") + _plot(zbp_name)).cols(ncols)


def score(
    zbp_ds: xr.Dataset, score_name: str, plunger_name: str = "p", field_name: str = "f"
) -> hv.DynamicMap:
    hv_ds = hv.Dataset(zbp_ds)
    return hv_ds.to(
        hv.Image,
        kdims=[field_name, plunger_name],
        vdims=[score_name],
        label=score_name,
        dynamic=True,
    ).options(aspect="square", cmap="inferno", tools=["hover"], colorbar=True)


def plot_with_discrete_cbar(
    ds, fig, ax, cmap_name="Greys", cbar_pad=0, label="", location="top", labelpad=7
):
    values = np.sort(np.unique(ds.data))
    ticks = np.linspace(0, 1, len(values))
    cmap = mpl.colors.ListedColormap(plt.cm.get_cmap(cmap_name)(ticks))
    im = ds.plot.imshow(ax=ax, add_colorbar=False, cmap=cmap)
    cbar = fig.colorbar(
        im,
        ax=ax,
        location=location,
        ticks=np.linspace(values.min(), values.max(), 2 * len(values) + 1)[1::2],
        pad=cbar_pad,
    )
    cbar.set_label(label, labelpad=labelpad)
    ticklabels = [f"{x:.2f}" for x in values]
    if location == "top":
        cbar.ax.set_xticklabels(ticklabels)
    else:
        cbar.ax.set_yticklabels(ticklabels)
    return im, cbar


def color_spline(ax, c, ls, lw=2, sides=["bottom", "top", "right", "left"]):
    for key in sides:
        ax.spines[key].set_color(c)
        ax.spines[key].set_linestyle(ls)
        ax.spines[key].set_linewidth(lw)
    ax.tick_params(axis="x", colors=c)
    ax.tick_params(axis="y", colors=c)
    for tick in ax.get_yticklabels():
        tick.set_color("k")
    for tick in ax.get_xticklabels():
        tick.set_color("k")
    ax.xaxis.set_tick_params(length=4, width=1, color="k")
    ax.yaxis.set_tick_params(length=4, width=1, color="k")
