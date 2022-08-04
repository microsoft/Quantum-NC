##
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MICROSOFT QUANTUM NON-COMMERCIAL License.
##
import logging
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import holoviews as hv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import display
from ipywidgets import Output, interactive
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from topogap_protocol.abstract_specs import (
    AbstractCalibrationData,
    AbstractPhaseOneData,
    AbstractPhaseTwoRegionData,
    AbstractRFData,
    _RequireAttrsABCMeta,
)
from topogap_protocol.clustering import (
    cluster_info,
    extract_clusters_local_2D,
    score_nonlocal_2D,
)
from topogap_protocol.datalake import (
    PathLike,
    datalake_fs,
    exists,
    load_dataset,
    path_or_url,
)
from topogap_protocol.gap_extraction import determine_gap_2D
# from topogap_protocol.rf_calibration import (
#     RFCalibration,
#     inverse_model,
#     pack_complex_coeffs,
# )
from topogap_protocol.symmetry import antisymmetric_conductance_part
from topogap_protocol.utils import (
    TMP_DIR,
    _holomap,
    broaden,
    drop_dimensionless_variables,
    is_homogeneous,
)

_LOGGER = logging.getLogger(__name__)


# class CalibrationData(AbstractCalibrationData, metaclass=_RequireAttrsABCMeta):
#     """Calibration data class."""
# 
#     data: Dict[Literal["left", "right"], xr.Dataset]
#     r_sq_threshold: float
#     coeffs: pd.DataFrame
#     rf_calibrations: Dict[Tuple[str, float], RFCalibration]
# 
#     def __init__(
#         self,
#         folder: Optional[PathLike] = None,
#         r_sq_threshold: float = 0.99,
#         do_bootstrap: bool = True,
#         *,
#         xr_datasets: Optional[Dict[Literal["left", "right"], xr.Dataset]] = None,
#     ):
#         if (folder is None and xr_datasets is None) or (
#             folder is not None and xr_datasets is not None
#         ):
#             raise ValueError("Specify either `folder` or `xr_datasets`.")
#         self.r_sq_threshold = r_sq_threshold
#         self.do_bootstrap = do_bootstrap
#         if folder is not None:
#             self.folder = path_or_url(folder)
#             self._folder = self.folder / "calibration"
#             self.load_data()
#         else:
#             self.folder = None
#             self.data = xr_datasets
#         self.check_data()
#         self.set_coeffs()
# 
#     def _data_per_side_and_field(self) -> Dict[Tuple[str, float], xr.Dataset]:
#         return {
#             (side, f): self.data[side].sel(f=f)
#             for side in ["left", "right"]
#             for f in self.data[side].f.values
#         }
# 
#     def _rf_calibrations(self) -> Dict[Tuple[str, float], RFCalibration]:
#         side_field_tuples, datasets = zip(*self._data_per_side_and_field().items())
#         with ProcessPoolExecutor() as ex:
#             cal = partial(
#                 RFCalibration,
#                 should_raise=False,
#                 r_sq_threshold=self.r_sq_threshold,
#                 do_bootstrap=self.do_bootstrap,
#             )
#             rfs = list(ex.map(cal, datasets))
#         return {(side, f): rf for (side, f), rf in zip(side_field_tuples, rfs)}
# 
#     def set_coeffs(self) -> None:
#         self.rf_calibrations = self._rf_calibrations()
#         coeffs = {
#             k: (*pack_complex_coeffs(rf.coeffs), rf.r_sq)
#             for k, rf in self.rf_calibrations.items()
#         }
#         df = pd.DataFrame(coeffs)
#         df.index = ["s0", "a", "b", "r_sq"]
# 
#         df.columns = pd.MultiIndex.from_tuples(coeffs.keys())
#         df = df.stack(dropna=False).unstack(
#             level=0
#         )  # Transposes the DataFrame one level deep
#         for side in ["left", "right"]:
#             with warnings.catch_warnings():
#                 warnings.filterwarnings("ignore", category=np.ComplexWarning)
#                 df = df.astype({(side, "r_sq"): float})
# 
#         # Extra check in case the calibration isn't good.
#         for side in ("left", "right"):
#             df_side = df[side]
#             pct_good = len(df_side[df_side.r_sq > self.r_sq_threshold]) / len(df_side)
#             if pct_good < 0.90:  # at least 90% of the values should make it.
#                 raise RuntimeError(
#                     f"The calibration R² at the {side} meets the threshold of"
#                     f" {self.r_sq_threshold*100:.0f}% in {pct_good*100:.0f}% of the cases."
#                     f" The mean R²={df_side.r_sq.mean():.3f}."
#                 )
#         self.coeffs = df
# 
#     def coeffs_at(self, side: str, f: float) -> Tuple[complex, complex, complex]:
#         df = self.coeffs[side].dropna()
#         fs = df.index
# 
#         def _interpolate(key):
#             vals = df[key].values
#             left, right = vals[0], vals[-1]
#             re = np.interp(f, fs, vals.real, left=left.real, right=right.real)
#             im = np.interp(f, fs, vals.imag, left=left.imag, right=right.imag)
#             return complex(re, im)
# 
#         s0, a, b = (_interpolate(k) for k in ["s0", "a", "b"])
#         return s0, a, b
# 
#     def plot_fit_results(self):
#         out = Output()
# 
#         def plot_fit_result(side, f) -> matplotlib.figure.Figure:
#             with out:
#                 out.clear_output(wait=True)
#                 rf = self.rf_calibrations[side, f]
#                 fig = rf.plot_model(show=False)
#                 fig.axes[-1].set_title(fr"{side}, $f={f}$ T, $R^2={rf.r_sq:#.4g}$")
#                 plt.show()
# 
#         widget = interactive(
#             plot_fit_result,
#             side=list(self.get_standard_xarray_names()),
#             f=list(self.coeffs.index.values),
#         )
#         return display(widget, out)
# 
#     def plot_goodness_of_fit(
#         self,
#         *,
#         show=False,
#         fname: Optional[Union[Path, str]] = None,
#     ) -> matplotlib.figure.Figure:
#         fs = self.coeffs.index
#         fig, axs = plt.subplots(nrows=2, sharex=True)
#         for ax, side in zip(axs, ["left", "right"]):
#             r_sq = self.coeffs[side]["r_sq"]
#             ax.set_ylim([min(0.989, np.min(r_sq)), 1])
#             ax.axhline(self.r_sq_threshold, c="k", lw=1)
#             ax.scatter(fs, r_sq, c={"left": "b", "right": "r"}[side])
#             ax.set_ylabel(f"$R^2$, {side}")
#             ax.grid(lw=0.25)
#         plt.tight_layout()
#         if show:
#             plt.show()
#         if fname is not None:
#             plt.savefig(fname, bbox_inches="tight")
#         return fig
# 
#     def plot_coefficients(
#         self,
#         *,
#         show=False,
#         fname: Optional[Union[Path, str]] = None,
#     ) -> matplotlib.figure.Figure:
#         fs = self.coeffs.index
#         fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(7, 6))
#         names = 2 * ["s0", "a", "b"]
#         sides = 3 * ["left"] + 3 * ["right"]
#         for (ax, name, side) in zip(axs.T.flatten(), names, sides):
#             if name == "s0":
#                 ax.set_title(f"{side.capitalize()}")
#             z = self.coeffs[side][name].values
#             plot_kwargs = dict(linestyle="-", marker=".")
#             ax.plot(fs, np.abs(z), linewidth=0.5, c="b", **plot_kwargs)
#             ax.set_ylabel(fr"$|{name}|$", c="b", fontsize=14)
#             ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#             _ax = ax.twinx()
#             z_arg = np.angle(z) / np.pi
#             _ax.plot(fs, z_arg, linewidth=0.25, c="g", **plot_kwargs)
#             _ax.set_ylabel(fr"arg$({name})/\pi$", c="g", fontsize=14)
#             if name == "b":
#                 ax.set_xlabel("Field [T]", fontsize=12)
#             ax.grid()
#         fig.suptitle(r"Fitting coefficients. $G = (s_0 + z)/(a s_0 - b z)$", y=1.02)
#         plt.tight_layout()
#         if show:
#             plt.show()
#         if fname is not None:
#             plt.savefig(fname, bbox_inches="tight")
#         return fig
# 
#     def nc_filename(self, side: str) -> Path:
#         return self._folder / f"{side}.nc"
# 
#     @property
#     def ncfiles(self) -> List[Path]:
#         return [self.nc_filename(s) for s in ("left", "right")]
# 
#     def ncfiles_complete(self) -> bool:
#         return all(exists(f) for f in self.ncfiles)
# 
#     def load_data(self) -> None:
#         sides = ("left", "right")
#         if not self.ncfiles_complete():
#             raise Exception(
#                 f"Need to extract data first, cannot find files in {self._folder}."
#             )
#         else:
#             self.data = {s: load_dataset(self.nc_filename(s)) for s in sides}


class _PhaseOneMixin(metaclass=_RequireAttrsABCMeta):
    @property
    def f_values(self):
        return next(iter(self.data.values())).f.values

    @property
    def side_combos(self):
        if self.cross_sweeps:
            return list(product("lr", repeat=2))
        else:
            return [("l", "l"), ("r", "r")]

    def nc_filename(self, c, b):
        return self._folder / f"{c}c_{b}b.nc"

    @property
    def ncfiles(self) -> List[Path]:
        return [self.nc_filename(c, b) for c, b in self.side_combos]

    def ncfiles_complete(self) -> bool:
        return all(exists(f) for f in self.ncfiles)

    def load_data(self) -> None:
        if not self.ncfiles_complete():
            raise Exception("Data is not complete!")

        self.data = {
            f"{c}c_{b}b": load_dataset(self.nc_filename(c, b))
            for c, b in self.side_combos
        }

        # Clean up single sized dimensions
        dims = set(self.get_standard_dim_names())
        for k, ds in self.data.items():
            to_drop = {
                name for name, size in ds.dims.items() if name not in dims and size == 1
            }
            if to_drop:
                print(f"Going to drop the dimensions {to_drop} from the dataset.")
            self.data[k] = drop_dimensionless_variables(ds.squeeze(to_drop, drop=True))

    def temperature_broaden(self, temperature=4.5e-6, which=("lc_lb", "rc_rb")):
        for key in which:
            bias_name = key.split("_")[1]
            self.data[key] = broaden(self.data[key], temperature, bias_name)


class RFData(_PhaseOneMixin, AbstractRFData):
    data: Dict[str, xr.Dataset]

    def __init__(
        self,
        folder: Optional[PathLike] = None,
        cross_sweeps=True,
        *,
        xr_datasets: Optional[Dict[str, xr.Dataset]] = None,
    ):
        if (folder is None and xr_datasets is None) or (
            folder is not None and xr_datasets is not None
        ):
            raise ValueError("Specify either `folder` or `xr_datasets`.")
        self.cross_sweeps = cross_sweeps
        if folder is not None:
            self.folder = path_or_url(folder)
            self._folder = self.folder / "rf"
            self.load_data()
        else:
            self.folder = None
            self.data = xr_datasets
        self.check_data()

    def add_magnitude(self):
        for name, ds in self.data.items():
            for side in "lr":
                ds[f"rf_{side}_mag"] = (
                    ds[f"rf_{side}_re"] ** 2 + ds[f"rf_{side}_im"] ** 2
                )

    def add_phase(self):
        for ds in self.data.values():
            for side in "lr":
                z = ds[f"rf_{side}_re"] + 1j * ds[f"rf_{side}_im"]
                ds[f"rf_{side}_phase"] = xr.apply_ufunc(np.angle, z, dask="allowed")

    def holomap_magnitude(self, name):
        self.add_magnitude()
        return _holomap(self.data, name, ("rf_l_mag", "rf_r_mag"))


def complex_signal_at(side, x):
    return x[f"rf_{side[0]}_re"] + 1j * x[f"rf_{side[0]}_im"]


class _ZBPMixin:
    def extract_zbp_dataset(
        self,
        threshold: float,
        bias_threshold: float = 1e-5,
        relative_prominence=0.04,
        relative_height=0.04,
        filter_params=dict(window_length=21, polyorder=2),
        peak_params=None,
        parallel: Union[bool, str] = "auto",
        combine_gaps="min",
    ):
        """
        Adds to data a new Dataset "zbp" containing
        ZBP maps for left and right sides, and a joint map indicating
        where ZBPs are present with a probability higher than a certain threshold.

        Parameters
        ----------
        threshold : float
            Minimum probability to pass the joint ZBP test.
        bias_threshold : float
            Maximum bias voltage for which a peak classifies as a zero bias peak
            for sim data set, the value is set to be 5x resolution in bias voltage.
        relative_prominence, relative_height, filter_params, peak_params
            See `find_peaks_or_dips_in_data_array`.
        parallel : Union[bool, Literal["auto"]]
            Call `find_peaks_or_dips_in_trace` in parallel. If "auto" the
            parallelization is enabled on Linux but disabled on Windows.
        combine_gaps : str
            How to combine gaps extracted from left and right. "min" uses minimum, "geometric" uses geometric mean

        Returns
        -------
        xarray.Dataset
            A dataset containing information about the postions of ZBPs.
        """
        from topogap_protocol.analysis import zbp_dataset

        zbp_ds = zbp_dataset(
            self,
            threshold,
            bias_threshold,
            relative_prominence,
            relative_height,
            filter_params,
            peak_params,
            parallel,
        )
        self._add_gap_to_zbp_ds(zbp_ds, combine_gaps)
        return ZBPDataset(zbp_ds)

    def extract_zbp_dataset_derivative(
        self,
        derivative_threshold: float,
        probability_threshold: float = 1.0,
        bias_window: float = 3e-2,
        polyorder: int = 2,
        average_over_cutter: bool = True,
        combine_gaps="min",
    ):
        """
        Adds to data a new Dataset "zbp" containing
        ZBP maps for left and right sides, and a joint map indicating
        where ZBPs are present with a probability higher than a certain threshold.

        Using the `scipy.signal.savgol_filter` to approximate the 2nd derivative.

        Parameters
        ----------
        phase_one_or_two_region : `PhaseOneData` or `PhaseTwoRegionData`
            A `PhaseOneData` or `PhaseTwoData` instance.
        derivative_threshold : float
            Minimum derivative per side.
        probability_threshold:
            Minimum probability to pass the joint ZBP test.
        bias_window, polyorder
            See `smooth_deriv_per_side`.
        average_over_cutter:
            Decides whether to average over cutter or not

        Returns
        -------
        xarray.Dataset
            A dataset containing information about the postions of ZBPs.
        """
        from topogap_protocol.analysis import zbp_dataset_derivative

        zbp_ds = zbp_dataset_derivative(
            self,
            derivative_threshold,
            probability_threshold,
            bias_window,
            polyorder,
            average_over_cutter,
        )
        self._add_gap_to_zbp_ds(zbp_ds, combine_gaps)
        return ZBPDataset(zbp_ds)

    def _add_gap_to_zbp_ds(self, zbp_ds, combine_gaps):
        if isinstance(self, PhaseTwoRegionData):
            if "gap" not in self.data["right"]:
                raise RuntimeError("First call `phase_two_region.extract_gap(...)`.")
            if combine_gaps == "min":
                zbp_ds["gap"] = np.minimum(
                    self.data["right"]["gap"].squeeze(),
                    self.data["left"]["gap"].squeeze(),
                )
            elif combine_gaps == "geometric":
                zbp_ds["gap"] = np.sqrt(
                    np.abs(
                        np.multiply(
                            self.data["right"]["gap"].squeeze(),
                            self.data["left"]["gap"].squeeze(),
                        )
                    )
                )
            elif combine_gaps == "left":
                zbp_ds["gap"] = self.data["left"]["gap"].squeeze()
            elif combine_gaps == "right":
                zbp_ds["gap"] = self.data["right"]["gap"].squeeze()
            else:
                raise RuntimeError(
                    "`combine_gaps` needs to be `min`, `geometric`, `left`, `right`"
                )


class PhaseOneData(_PhaseOneMixin, _ZBPMixin, AbstractPhaseOneData):
    data: Dict[str, xr.Dataset]

    def __init__(
        self,
        folder: Optional[PathLike] = None,
        cross_sweeps=True,
        *,
        xr_datasets: Optional[
            Dict[str, xr.Dataset]
        ] = None,
    ):
        if (folder is None and xr_datasets is None) or (
            folder is not None and xr_datasets is not None
        ):
            raise ValueError("Specify either `folder` or `xr_datasets`.")
        self.cross_sweeps = cross_sweeps
        if folder is not None:
            self.folder = path_or_url(folder)
            self._folder = self.folder / "g"
            if not exists(self._folder):
                raise OSError(
                    f"Could not locate PhaseOneData folder {self._folder}."
                    " Use `extract_local_conductance` to create the conductance"
                    " data because it hasn't been created."
                )
            self.load_data()
        else:
            self.folder = None
            self.data = xr_datasets
        self.check_data()

    def holomap_conductance(self, name):
        return _holomap(self.data, name, ("g_ll", "g_rr"))

    def plot_interactive(
        self,
        norm: Optional[matplotlib.colors.Normalize] = None,
        b_vs_f: bool = True,
        p_vs_f: bool = True,
        p_vs_b: bool = True,
    ) -> interactive:
        from topogap_protocol.plot import phase_one_interactive

        return phase_one_interactive(self, norm, b_vs_f, p_vs_f, p_vs_b)

    def interpolate_to_square_grid(self, n: Optional[int] = None, n_max: int = 200):
        """Interpolate the plunger and field values onto a finer square grid.

        Parameters
        ----------
        n : Optional[int]
            Fixed number of points in 'f' and 'p'. If None, 'n' is automatically
            determined, based on ``max(len(ds.f), len(ds.p))``.
        n_max : int
            The maximal number of points in 'f' and 'p'. This is limited to avoid memory
            issues.
        """
        for key in self.data:
            ds = self.data[key]
            n = max(len(ds.f), len(ds.p))
            n = min(n, n_max)
            to_interp = {}
            if len(ds.p) != n or not is_homogeneous(ds.p):
                to_interp["p"] = np.linspace(ds.p[0], ds.p[-1], n)
            if len(ds.f) != n or not is_homogeneous(ds.f):
                to_interp["f"] = np.linspace(ds.f[0], ds.f[-1], n)
            self.data[key] = ds.interp(**to_interp)


class PhaseTwoRegionData(
    AbstractPhaseTwoRegionData, _ZBPMixin, metaclass=_RequireAttrsABCMeta
):
    region_number: int
    folder: Optional[PathLike]

    def __init__(
        self,
        folder: Optional[PathLike] = None,
        region: int = 0,
        *,
        xr_datasets: Optional[Dict[str, xr.Dataset]] = None,
    ) -> None:
        if (folder is None and xr_datasets is None) or (
            folder is not None and xr_datasets is not None
        ):
            raise ValueError("Specify either `folder` or `xr_datasets`.")
        self.region_number = region
        if folder is not None:
            self.folder = path_or_url(folder)
            self._folder = folder / f"{region:02d}"
            self.load_data()
        else:
            self.folder = None
            self.data = xr_datasets
        self.check_data()

    def nc_filename(self, side):
        return self._folder / f"{side}.nc"

    @property
    def ncfiles(self) -> List[Path]:
        return [self.nc_filename(s) for s in ("left", "right")]

    def ncfiles_complete(self) -> bool:
        return all(exists(f) for f in self.ncfiles)

    def load_data(self) -> None:
        if not self.ncfiles_complete():
            raise Exception("Need to extract data first")
        self.data = {s: load_dataset(self.nc_filename(s)) for s in ("left", "right")}

    def extract_gap(
        self,
        median_size=2,
        gauss_sigma=(1, 2),
        threshold_factor=0.1,
        shift=True,
    ):
        """Adds the gap to all regions and sides for a ``phase_two`` instance.

        Parameters
        ----------
        median_size : scalar or tuple, optional
            Size of the median filter, ``size`` argument to `scipy.ndimage.median_filter`.
        gauss_sigma : scalar or sequence of scalars
            Size of the gaussian filter, ``sigma`` argument to `scipy.ndimage.gaussian_filter`.
        threshold_factor : float
            Threshold is `threshold_factor * max(array)`.
        shift : bool, optional
            Whether to shift bias for antisymmetric part of conductance
        """
        # merge gap data with xarray of left and right side
        for side in ("left", "right"):
            other_side = {"left": "r", "right": "l"}[side]
            data = self.data[side]
            key = f"g_{other_side}{side[0]}"
            (gap, filtered_antisym_g) = determine_gap_2D(
                data[key],
                bias_name=f"{side[0]}b",
                median_size=median_size,
                gauss_sigma=gauss_sigma,
                threshold_factor=threshold_factor,
                filtered_antisym_g=True,
                shift=shift,
            )
            self.data[side] = data.assign(
                gap=gap, filtered_antisym_g=filtered_antisym_g
            )
            self.data[side].filtered_antisym_g.attrs["median_size"] = median_size
            self.data[side].filtered_antisym_g.attrs["gauss_sigma"] = gauss_sigma
            self.data[side].gap.attrs["threshold_factor"] = threshold_factor

    def plot_gap_extraction(
        self,
        cutter_value,
        plunger_value,
        plot_filtered_conductance=True,
        fname=None,
        show=True,
    ) -> matplotlib.figure.Figure:
        fig, axs = plt.subplots(ncols=2, figsize=(8, 6), sharex=True, sharey=True)
        cut_left = self.data["left"].sel(
            {"lc": cutter_value, "p": plunger_value}, method="nearest"
        )
        if plot_filtered_conductance:
            g_rl = cut_left.filtered_antisym_g
        else:
            g_rl = cut_left.g_rl.squeeze().transpose("f", "lb")
            g_rl.data = antisymmetric_conductance_part(
                cond=g_rl.data.T, bias=cut_left["lb"].data
            ).T

        _ = (
            g_rl.squeeze()
            .transpose("lb", "f")
            .plot.imshow(ax=axs[0], add_colorbar=False, cmap="seismic")
        )
        cut_right = self.data["right"].sel(
            {"rc": cutter_value, "p": plunger_value}, method="nearest"
        )
        if plot_filtered_conductance:
            g_lr = cut_right.filtered_antisym_g
        else:
            g_lr = cut_right.g_lr.squeeze().transpose("f", "rb")
            g_lr.data = antisymmetric_conductance_part(
                cond=g_lr.data.T, bias=cut_right["rb"].data
            ).T
        im_right = (
            g_lr.squeeze()
            .transpose("rb", "f")
            .plot.imshow(ax=axs[1], add_colorbar=False, cmap="seismic")
        )
        fig.colorbar(
            im_right,
            ax=axs.ravel().tolist(),
            label="antisym. $G_{LR,RL}$ [$e^2/h$]",
        )
        cut_left.gap.plot(ax=axs[0])
        cut_right.gap.plot(ax=axs[1])

        axs[0].set_title(r"$G_{LR}$ and $\Delta_\mathrm{ex}$ left")
        axs[1].set_title(r"$G_{RL}$ and $\Delta_\mathrm{ex}$ right")
        which = "Filtered" if plot_filtered_conductance else "Unfiltered"
        fig.suptitle(f"{which} antisymmetric part of $G$", fontsize=16)
        if show:
            plt.show()

        if fname is not None:
            plt.savefig(fname, bbox_inches="tight")
        return fig

    def plot_gap_extraction_interactive(self) -> None:
        out = Output()

        def plot(
            cutter_value, plunger_value, plot_filtered_conductance, with_2d_plot
        ) -> matplotlib.figure.Figure:
            with out:
                out.clear_output(wait=True)
                self.plot_gap_extraction(
                    cutter_value, plunger_value, plot_filtered_conductance
                )
                if with_2d_plot:
                    self.plot_extracted_gap(cutter_value, plunger_hline=plunger_value)
                plt.show()

        # TODO: It might be that the left and right data are taken for
        # different cutter values. It that is the case, change the function
        # to take `lc`, `rc` instead of `cutter_value`.
        widget = interactive(
            plot,
            cutter_value=list(self.data["left"].lc.values),
            plunger_value=list(self.data["left"].p.values),
            plot_filtered_conductance=[True, False],
            with_2d_plot=[True, False],
        )
        return display(widget, out)

    def plot_extracted_gap(
        self,
        cutter_value,
        plunger_hline: Optional[float] = None,
        fname: Optional[Union[Path, str]] = None,
        show: bool = True,
    ) -> matplotlib.figure.Figure:
        fig, axs = plt.subplots(ncols=2, constrained_layout=True, figsize=(5, 5))
        for side, ax in zip(["left", "right"], axs):
            im_gap = (
                self.data[side]["gap"]
                .sel(rc=cutter_value, lc=cutter_value, method="nearest")
                .squeeze()
                .plot.imshow(
                    ax=ax,
                    add_colorbar=False,
                    cmap="gist_heat_r",
                    vmin=0,
                )
            )
            g = {"left": "G_{RL}", "right": "G_{LR}"}[side]
            ax.set_title(f"Gap from ${g}$")
            cbar = fig.colorbar(im_gap, ax=ax, location="top", extend="max")
            cbar.set_label(r"extracted gap $\Delta_\mathrm{ex}$")
            if plunger_hline is not None:
                ax.axhline(plunger_hline, ls="--", lw=1, c="k")
        if fname is not None:
            plt.savefig(fname, bbox_inches="tight", transparent=True)
        if show:
            plt.show()
        return fig

    def plot_interactive(
        self,
        norm: Optional[matplotlib.colors.Normalize] = None,
        unit: Optional[str] = None,
    ) -> interactive:
        from topogap_protocol.plot import phase_two_interactive

        return phase_two_interactive(self, norm, unit)


class PhaseTwoData(metaclass=_RequireAttrsABCMeta):
    regions: Dict[int, PhaseTwoRegionData]

    def __init__(
        self,
        folder: Optional[PathLike] = None,
        *,
        regions: Optional[List[PhaseTwoRegionData]] = None,
    ):
        if (folder is None and regions is None) or (
            folder is not None and regions is not None
        ):
            raise ValueError("Specify either `folder` or `regions`.")
        if folder is not None:
            self.folder = path_or_url(folder)
            self._folder = self.folder / "phase_two"
            self.load_data()
        else:
            self.regions = {region.region_number: region for region in regions}

    def extract_gap(self, median_size=2, gauss_sigma=(1, 2), threshold_factor=0.1):
        """Adds the gap to all regions and sides for a ``phase_two`` instance.

        Parameters
        ----------
        median_size : scalar or tuple, optional
            Size of the median filter.
        gauss_sigma : scalar or sequence of scalars
            Size of the gaussian filter.
        threshold_factor : float
            Threshold is `threshold_factor * max(array)`.
        """
        for _, region in self.regions.items():
            region.extract_gap(median_size, gauss_sigma, threshold_factor)

    def load_data(self) -> None:
        if isinstance(self._folder, Path):
            fnames = os.listdir(self._folder)
        else:
            fs = datalake_fs()
            listdir = fs.ls(str(self._folder).replace(r"abfs://", ""))
            fnames = [Path(f).name for f in listdir]
        regions = [int(f) for f in fnames if re.search(r"^\d{2}$", f)]
        self.regions = {i: PhaseTwoRegionData(self._folder, i) for i in regions}

    def __str__(self):
        s = (
            "Phase two data of the topological gap protocol.",
            "",
            f"Device ID         : {self.device_id}",
            f"Number of regions : {len(self.regions)}",
        )
        return "\n".join(s)


# def extract_local_conductance(
#     calibration_data: CalibrationData,
#     rf_data: RFData,
#     save_or_upload: bool = True,
#     return_phase_one: bool = False,
# ) -> Optional[PhaseOneData]:
#     datas: Dict[Literal["rc_rb", "rc_lb", "lc_rb", "lc_lb"], xr.Dataset] = {}
#     for c, b in rf_data.side_combos:
#         name = f"{c}c_{b}b"
#         ds = rf_data.data[name]
#         datasets = []
#         for i, f in enumerate(rf_data.f_values):
#             gs = {}
#             sides = (
#                 ("left", "right")
#                 if rf_data.cross_sweeps
#                 else [{"r": "right", "l": "left"}[c]]
#             )
#             for side in sides:
#                 x = ds.sel(f=f, method="nearest")
#                 s = complex_signal_at(side, x)
#                 g = inverse_model(s, *calibration_data.coeffs_at(side, f)).real
#                 g = xr.DataArray(g, coords=s.coords, dims=s.dims).expand_dims("f")
#                 letter = side[0].upper()
#                 g.attrs = {
#                     "long_name": f"dI_{letter}/dV_{letter}",
#                     "units": "e^2/h",
#                 }
#                 gs[side] = g
# 
#             if rf_data.cross_sweeps:
#                 conductance = gs["left"].to_dataset(name="g_ll")
#                 conductance["g_rr"] = gs["right"]
#             else:
#                 conductance = gs[side].to_dataset(name=f"g_{side[0]}{side[0]}")
#             datasets.append(conductance)
#         ds = xr.concat(datasets, dim="f").load()
#         datas[name] = ds
#         if save_or_upload:
#             if rf_data.folder is None:
#                 raise ValueError(
#                     "`save_or_upload` can only be true if `rf_data` is stored locally."
#                 )
#             fname = rf_data.folder / "g" / f"{name}.nc"
#             if isinstance(fname, Path):
#                 fname.parent.mkdir(parents=True, exist_ok=True)
#                 ds.to_netcdf(fname)
#             else:  # isinstance(fname, URL) == True
#                 tmp_fname = TMP_DIR / fname.name
#                 tmp_fname.parent.mkdir(parents=True, exist_ok=True)
#                 ds.to_netcdf(tmp_fname)
#                 print(f"Uploading to DataLake under {fname}")
#                 fs = datalake_fs()
#                 fs.put(str(tmp_fname), str(fname))
#         if not return_phase_one:
#             ds.close()
# 
#     if return_phase_one:
#         return PhaseOneData(xr_datasets=datas, cross_sweeps=rf_data.cross_sweeps)


class ZBPDataset:
    def __init__(self, xr_dataset):
        self.data = xr_dataset
        self.data["f"].attrs["long_name"] = "$B$"
        self.data["f"].attrs["units"] = "T"
        self.data["p"].attrs["long_name"] = "$V_p$"
        self.data["p"].attrs["units"] = "$V$"

    def set_gap_threshold(self, threshold_low=10e-6, threshold_high=None) -> None:
        """Set gap threshold to an energy that should be considered 'gapped'.

        Adds both 'gap_boolean' and 'gapped_zbp' to 'self.data'.
        """
        self.data["gap_boolean"] = self.data["gap"] > threshold_low
        self.data["gapped_zbp"] = self.data["gap_boolean"] * self.data["zbp"] > 0
        if threshold_high is not None:
            self.data["gapped_zbp"] *= self.data["gap"] < threshold_high

    def set_boundary_array(
        self,
        pixel_size=3,
        name_cluster="zbp_cluster",
        name_boundary_indices="boundary_indices",
        name_boundary_array="boundary_array",
        all_boundaries=False,
    ):
        def _add_boundary_array(cluster_array, boundary, pixel_size):

            boundary_arr = np.zeros_like(cluster_array) * np.nan
            for (i, j), is_topo in boundary.items():
                if is_topo or all_boundaries:
                    px = pixel_size - 1
                    boundary_arr[i - px : i + px + 1, j - px : j + px + 1] = 1
            return boundary_arr

        self.data[name_boundary_array] = xr.apply_ufunc(
            _add_boundary_array,
            self.data[name_cluster],
            self.data[name_boundary_indices],
            input_core_dims=[["p", "f"], []],
            output_core_dims=[["p", "f"]],
            vectorize=True,
            kwargs=dict(pixel_size=pixel_size),
        )

    def extract_clusters(
        self,
        name_zbp="zbp",
        min_cluster_size=350,
        xi=0.1,
        max_eps=10,
        min_samples=10,
    ):
        unwanted_dims = [
            (k, v.size)
            for k, v in self.data[name_zbp].coords.items()
            if k not in ("p", "f") and v.size > 1
        ]
        if unwanted_dims:
            raise ValueError(
                "Cluster extraction needs to happen on an array with "
                "dimensions 'p' and 'f'. "
                f"Reduce the dimensions {unwanted_dims} in 'self.data['{name_zbp}'] "
                "or provide a variable with the correct dimensions."
            )
        # extract_clusters_local_2D cannot act inplace when the function
        # has been run before, so we overwrite 'self.data'.
        self.data = extract_clusters_local_2D(
            self.data,
            name_zbp=name_zbp,
            min_cluster_size=min_cluster_size,
            xi=xi,
            max_eps=max_eps,
            min_samples=min_samples,
            cluster_method="xi",
        )

    def get_cluster_infos(
        self,
        pct_box=20,
        name_cluster="zbp_cluster",
        name_zbp_cluster_number="zbp_cluster_number",
    ) -> Dict[int, Dict[str, Any]]:
        infos = {}
        for i in self.data[name_zbp_cluster_number]:
            cluster = self.data[name_cluster].sel(**{name_zbp_cluster_number: i})
            info = cluster_info(cluster, "p", "f", pct_box)
            infos[int(i)] = info
        return infos

    def get_region_of_interest_1(
        self,
        pct_box: int = 20,
        min_area: float = 0.01,
        name_cluster="zbp_cluster",
        name_zbp_cluster_number="zbp_cluster_number",
        zbp_cluster_number=0,
    ) -> Tuple[Dict[str, Any], xr.DataArray]:
        infos = self.get_cluster_infos(pct_box)
        large_enough_areas = [
            (info["center"][1], i)
            for i, info in infos.items()
            if info["area"] > min_area and i != -1
        ]
        i = sorted(large_enough_areas)[zbp_cluster_number][1]  # lowest one
        return (
            infos[i],
            self.data[name_cluster].sel(**{name_zbp_cluster_number: i}),
        )

    def set_score(
        self,
        name_cluster: str = "zbp_cluster",
        name_gap: str = "gap_boolean",
        name_gap_value: str = "gap",
        name_average_score: str = "nonlocal_score_average",
        name_median_score: str = "nonlocal_score_median",
        name_max_score: str = "nonlocal_score_max",
        name_percentage_boundary: str = "percentage_boundary",
        name_median_gap: str = "median_gap",
        name_average_gap: str = "average_gap",
        name_boundary_indices: str = "boundary_indices",
        variance: int = 5,
    ) -> None:
        """Adds local nonscore data to the clusters, based on average
        and median gap in the cluster, and percentage of the boundary
        that is gapless.

        Parameters
        ----------
        name_cluster : str, optional
            Name for the new column corresponding to the boolean cluster positions, by default "zbp_cluster".
        name_gap : str, optional
            Name of name_gap in ``data_2D``, by default "gap_boolean".
        name_gap_value : str, optional
            Name of name_gap_value in ``data_2D``, by default "gap".
        name_average_score : str, optional
            New name of name_average_score in ``data_2D``, by default "nonlocal_score_average".
        name_median_score : str, optional
            New name of name_median_score in ``data_2D``, by default "nonlocal_score_median".
        name_max_score : str, optional
            New name of name_max_score in ``data_2D``, by default "nonlocal_score_max".
        variance : int, optional
            Position tolerance for distance between the ZBP array end and the gap
            closing, by default 5.

        Returns
        -------
        None
        """
        score_nonlocal_2D(
            self.data,
            "f",
            "p",
            name_cluster,
            name_gap,
            name_gap_value,
            name_average_score,
            name_median_score,
            name_max_score,
            name_percentage_boundary,
            name_median_gap,
            name_average_gap,
            name_boundary_indices,
            variance,
        )

    def plot_clusters(
        self,
        pct_box=5,
        min_area=0.01,
        zbp_cluster_number=0,
        fig=None,
        ax=None,
        fname: Optional[Union[Path, str]] = None,
    ):
        from topogap_protocol.analysis import flatten_clusters

        ds = self.data
        if ax is None:
            fig, ax = plt.subplots(constrained_layout=True)
        clen = len(ds["zbp_cluster_number"])
        colors = ["white"]
        ticklabels = ["bg"]
        if -1 in ds["zbp_cluster_number"]:
            colors.append("black")
            ticklabels.append("no")
        colors += list(plt.cm.get_cmap("Set1").colors) + list(
            plt.cm.get_cmap("Set2").colors
        )
        colors = colors[0 : (clen + 1)]

        cmap = matplotlib.colors.ListedColormap(colors)
        clusters = flatten_clusters(ds, "p", "f", "zbp_cluster").T
        im2 = clusters.plot.imshow(
            ax=ax,
            cmap=cmap,
            add_colorbar=False,
        )
        ax.set_title("")
        cluster_ticks = np.array(range(-1, clen, 1))
        cbar = fig.colorbar(
            im2,
            ax=ax,
            location="top",
            ticks=clen * (cluster_ticks + 1.5) / (clen + 1),
            pad=0,
        )
        cbar.set_label("Cluster ID")
        ticklabels.extend(range(1, len(ds["zbp_cluster_number"]) + 1))
        cbar.ax.set_xticklabels(ticklabels[: len(cluster_ticks)])

        # Create a Rectangle patch on the selected cluster
        try:
            info, selected_cluster = self.get_region_of_interest_1(
                pct_box, min_area, zbp_cluster_number=zbp_cluster_number
            )
            x1, y1, x2, y2 = info["bounding_box"]
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    (x2 - x1),
                    (y2 - y1),
                    linewidth=2,
                    edgecolor=colors[zbp_cluster_number + 2],
                    facecolor="none",
                )
            )
        except IndexError:
            selected_cluster = None
            print("Didn't find any clusters.")

        if fname is not None:
            plt.savefig(fname, transparent=True)

        return selected_cluster

    def plot_joined_probability(
        self,
        threshold: Optional[float] = None,
        zbp_name: str = "zbp",
        figsize: Tuple[float, float] = (5.1, 3.15),
        title=None,
        fname=None,
        show=True,
    ):
        from topogap_protocol.plot import joined_zbp_probability

        joined_zbp_probability(
            self.data,
            threshold or self.data.attrs.get("threshold", "probability_threshold"),
            "p",
            "f",
            zbp_name,
            figsize,
            title,
            fname,
            show,
        )

    def plot_left_right_probability(
        self,
        threshold: Optional[float] = None,
        figsize: Tuple[float, float] = (5.1, 3.15),
        discrete=False,
        fname=None,
    ) -> matplotlib.figure.Figure:
        from topogap_protocol.plot import left_right_zbp_probability

        return left_right_zbp_probability(
            self.data,
            threshold or self.data.attrs.get("threshold", "probability_threshold"),
            "p",
            "f",
            figsize,
            discrete,
            fname,
        )

    def plot_probability_and_clusters(
        self,
        pct_box=10,
        fname=None,
        with_clusters=True,
        min_area=0.01,
    ) -> matplotlib.figure.Figure:
        """Based on the paper's Fig. 6."""
        from topogap_protocol.plot import plot_with_discrete_cbar

        single_column_width = 3 + 3 / 8
        fig = plt.figure(
            constrained_layout=True,
            figsize=(2 * single_column_width, 2 * single_column_width),
        )
        gs = GridSpec(2, 2, figure=fig)

        ds = self.data

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1], sharey=ax0, sharex=ax0)
        ax2 = fig.add_subplot(gs[1, 0], sharey=ax0, sharex=ax0)
        axs = [ax0, ax1, ax2]
        if with_clusters:
            ax3 = fig.add_subplot(gs[1, 1])
            axs.append(ax3)

        # ZBP probabilities
        for key, which, ax in zip(
            ("left", "right", "zbp"),
            ("Left", "Right", "Joint"),
            [ax0, ax1, ax2],
        ):
            plot_with_discrete_cbar(
                ds[key].squeeze().transpose("p", "f"),
                fig,
                ax,
                "viridis",
                0,
                f"{which} ZBP Probability",
            )

        # Clusters
        if with_clusters:
            self.plot_clusters(pct_box, min_area, fig, ax3)

        for i, ax in enumerate(axs):
            ax.set_title("")
            label = "abcd"[i]
            ax.text(
                0.15,
                0.95,
                fr"$\mathrm{{({label})}}$",
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=16,
                color="white" if i != 3 else "black",
            )
        if fname is not None:
            plt.savefig(fname, bbox_inches="tight", transparent=True)
        plt.show()
        return fig

    def plot_extracted_gap(
        self,
        cutter_value,
        fname: Optional[Union[Path, str]] = None,
        show: bool = True,
    ) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))
        sel = {}
        for c in ("rc", "lc"):
            if c in self.data["gap"].dims:
                sel[c] = cutter_value
        im_gap = (
            self.data["gap"]
            .sel(**sel, method="nearest")
            .squeeze()
            .plot.imshow(
                ax=ax,
                add_colorbar=False,
                cmap="gist_heat_r",
                vmin=0,
            )
        )
        cbar = fig.colorbar(im_gap, ax=ax, location="top", extend="max")
        cbar.set_label(r"extracted gap $\Delta_\mathrm{ex}$")
        ax.set_title("")
        if fname is not None:
            plt.savefig(fname, bbox_inches="tight", transparent=True)
        if show:
            plt.show()
        return fig

    def plot_region_of_interest_2(
        self,
        cutter_value: float,
        zbp_cluster_number: int,
        fname: Optional[Union[Path, str]] = None,
        show: bool = True,
        boundary=True,
        padding=1,
        gap_units=" [meV]",
    ) -> matplotlib.figure.Figure:
        from topogap_protocol.plot import color_spline

        fig, axs = plt.subplots(ncols=2, constrained_layout=True, figsize=(8, 6))
        sel = {}
        for c in ("rc", "lc"):
            if c in self.data["gap"].dims:
                sel[c] = cutter_value
        cluster = self.data.sel(**sel, zbp_cluster_number=zbp_cluster_number)
        gap_bool = self.data.gap_boolean.sel(**sel).squeeze()
        zbp_bool = self.data.zbp.squeeze()

        ds = (2 / 4) * gap_bool + (1 / 4) * zbp_bool + (1 / 4) * cluster.zbp_cluster
        ds = ds.squeeze().transpose("p", "f")
        values = np.sort(np.unique(ds.data))
        cmap = matplotlib.colors.ListedColormap(
            list(plt.cm.Greys(np.linspace(0, 1, len(values) - 1))) + ["red"]
        )
        im = ds.plot.imshow(
            ax=axs[0],
            add_colorbar=False,
            cmap=cmap,
        )
        axs[0].set_title("")
        cbar = fig.colorbar(
            im,
            ax=axs[0],
            location="top",
            ticks=np.linspace(values.min(), values.max(), 2 * len(values) + 1)[1::2],
        )
        cbar.set_label("Gap and ZBP regions")
        ticks_options = {
            0: "gpls",
            0.25: "gpls ZBP",
            0.5: "gap",
            0.75: "gap ZBP",
            1: "ROI$_2$",
        }
        ticklabels = [ticks_options[x] for x in values]
        cbar.ax.set_xticklabels(ticklabels, fontsize=10)
        if boundary:
            for _zbp_cluster_number in self.data.zbp_cluster_number[1:]:
                (
                    self.data["boundary_array"]
                    .sel(**sel)
                    .sel(zbp_cluster_number=_zbp_cluster_number)
                    .squeeze()
                    .transpose("p", "f")
                    .plot.imshow(
                        ax=axs[0],
                        add_colorbar=False,
                        cmap="ocean",
                    )
                )
        axs[0].set_title("")

        first, *_, last = np.where(cluster.zbp_cluster.sum("f") > 0)[0]

        dashed = (
            max(cluster.p.min(), cluster.p[first - padding]),
            min(cluster.p.max(), cluster.p[last + padding]),
        )
        first_f, *_, last_f = np.where(cluster.zbp_cluster.sum("p") > 0)[0]

        f_range = (
            max(cluster.f.min(), cluster.f[max(first_f - padding, 0)]),
            min(
                cluster.f.max(),
                cluster.f[min(last_f + padding, len(cluster.f) - padding)],
            ),
        )
        for y in dashed:
            axs[0].axhline(y, ls="--", lw=1, c="k")

        cluster_gap = (
            (self.data.gap * self.data.zbp_cluster)
            .sel(zbp_cluster_number=zbp_cluster_number)
            .sel(**sel)
            .squeeze()
        )

        cluster_gap.transpose("p", "f").plot.imshow(
            ax=axs[1],
            add_colorbar=True,
            cmap="gist_heat_r",
            cbar_kwargs={
                "label": r"Gap inside ROI$_2$" + gap_units,
                "location": "top",
            },
        )

        median_gap = float(cluster.median_gap.squeeze())
        pct_boundary = 100 * float(cluster.percentage_boundary.squeeze())

        axs[1].text(
            0.1,
            0.15,
            r"$\bar{\Delta}_\mathrm{ex}^{(j)}="
            + rf"{median_gap:.4f}$"
            + gap_units
            + "\n"
            + fr"$\mathrm{{gapless boundary}}={pct_boundary:.0f}$ %",
            transform=axs[1].transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=13,
            color="k",
        )

        axs[1].yaxis.set_major_locator(plt.MaxNLocator(3))

        axs[1].set_ylim(*dashed)
        axs[1].set_xlim(*f_range)
        axs[1].set_title("")
        color_spline(axs[1], "k", "--", 1, ["bottom", "top"])

        for i, ax in enumerate(axs):
            label = "abcd"[i]
            color = {0: "black", 1: "black", 2: "black", 3: "black"}[i]
            ax.text(
                -0.1,
                1,
                fr"$\mathrm{{({label})}}$",
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=14,
                color=color,
            )

        if fname is not None:
            plt.savefig(fname, bbox_inches="tight", transparent=True)
        if show:
            plt.show()
        return fig

    def plot_zbp_per_side_and_conductance_cuts(self, phase_one, f, p, fname=None):
        """Based on the paper's Fig. 6."""
        from topogap_protocol.plot import plot_with_discrete_cbar

        single_column_width = 3 + 3 / 8
        fig = plt.figure(
            constrained_layout=True,
            figsize=(2 * single_column_width, 2 * single_column_width),
        )
        gs = GridSpec(2, 2, figure=fig)

        ds = self.data

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1], sharey=ax0, sharex=ax0)
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1], sharey=ax2, sharex=ax2)
        axs = [ax0, ax1, ax2, ax3]

        # ZBP probabilities
        for key, which, ax in zip(
            ("left", "right"),
            ("Left", "Right"),
            [ax0, ax1],
        ):
            plot_with_discrete_cbar(
                ds[key].squeeze().transpose("p", "f"),
                fig,
                ax,
                "viridis",
                0,
                f"{which} ZBP Probability",
            )
            ax.scatter([f], [p], s=100, c="r", marker="x")

        for k, c, g, ax in [("lc_lb", "lc", "g_ll", ax2), ("rc_rb", "rc", "g_rr", ax3)]:
            side = phase_one.data[k].sel(f=f, p=p, method="nearest")
            cutters = side[c].values
            N = len(cutters)
            cmap = matplotlib.cm.get_cmap("jet", N)
            norm = matplotlib.colors.BoundaryNorm(np.arange(N + 1) + 0.5, N)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            for i, cutter_value in enumerate(cutters):
                sel = side.sel(**{c: cutter_value})
                sel[g].squeeze().plot.line(ax=ax, c=cmap(i / N))
            s = k[0] + k[0]
            cbar = fig.colorbar(
                sm,
                ax=ax,
                location="top",
                ticks=np.arange(N + 1)[1:],
                label=rf"$G_{{\mathrm{{{s}}}}}$ line cut per cutter",
            )
            ticklabels = [f"{x:.2f}" for x in cutters]
            cbar.ax.set_xticklabels(ticklabels)

        for i, ax in enumerate(axs):
            ax.set_title("")
            label = "abcd"[i]
            ax.text(
                0.20,
                0.95,
                fr"$\mathrm{{({label})}}$",
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=16,
                color="white" if i < 2 else "black",
            )
        if fname is not None:
            plt.savefig(fname, bbox_inches="tight", transparent=True)
        plt.show()

    def plot_zbp_per_side_and_conductance_cuts_interactive(self, phase_one):
        out = Output()

        def _plot(f, p) -> matplotlib.figure.Figure:
            with out:
                out.clear_output(wait=True)
                self.plot_zbp_per_side_and_conductance_cuts(phase_one, f, p)
                plt.show()

        widget = interactive(
            _plot,
            p=self.data.p.values.tolist(),
            f=self.data.f.values.tolist(),
        )
        display(widget, out)

    def plot_gapped_zbp(self, cutter_value):
        from topogap_protocol.plot import plot_with_discrete_cbar

        figsize = (8, 8)
        fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        sel = {}
        for c in ("rc", "lc"):
            if c in self.data["gap"].dims:
                sel[c] = cutter_value
        self.data["gap"].sel(**sel).plot.imshow(ax=axs[0, 0], vmin=0)
        axs[0, 0].set_title(r"Extracted gap $\Delta_{ex}$")

        gap_boolean = self.data["gap_boolean"].sel(**sel)
        _, cbar = plot_with_discrete_cbar(
            gap_boolean,
            fig,
            axs[0, 1],
            "Greys",
            cbar_pad=0,
            label="",
            location="right",
        )
        axs[0, 1].set_title("Gap boolean")
        ticks = []
        if False in gap_boolean.values:
            ticks.append("Gapless")
        if True in gap_boolean.values:
            ticks.append("Gapped")
        cbar.ax.set_yticklabels(ticks)

        _, cbar = plot_with_discrete_cbar(
            self.data["zbp"].transpose(),
            fig,
            axs[1, 0],
            "Greys",
            cbar_pad=0,  # -0.22,
            label="",
            location="right",
        )
        axs[1, 0].set_title("ZPB boolean")
        cbar.ax.set_yticklabels(["no ZBP", "ZBP"][: len(cbar.ax.get_yticks())])

        _, cbar = plot_with_discrete_cbar(
            self.data["gapped_zbp"].sel(**sel),
            fig,
            axs[1, 1],
            "Greys",
            cbar_pad=0,
            label="",
            location="right",
        )
        axs[1, 1].set_title("Gapped ZBP")
        cbar.ax.set_yticklabels(["Gapless", "Gapped ZBP"][: len(cbar.ax.get_yticks())])
        return fig

    def plot_gapped_zbp_interactive(self) -> None:
        out = Output()

        def _plot(
            cutter_value, threshold_low, threshold_high
        ) -> matplotlib.figure.Figure:
            with out:
                self.set_gap_threshold(threshold_low, threshold_high)
                out.clear_output(wait=True)
                self.plot_gapped_zbp(cutter_value)
                plt.show()

        gaps = sorted(np.unique(self.data["gap"].values.flat))
        widget = interactive(
            _plot,
            cutter_value=np.atleast_1d(self.data.rc.values.tolist()),
            threshold_low=gaps,
            threshold_high=[None] + gaps,
        )
        display(widget, out)

    def plot_probabilities_holoviews(
        self, zbp_name: str = "zbp", ncols: int = 2
    ) -> hv.Layout:
        from topogap_protocol.plot import zbps

        return zbps(self.data, ncols=ncols, zbp_name=zbp_name)

    def plot_score_holoviews(self, score_name: str) -> hv.DynamicMap:
        from topogap_protocol.plot import score

        return score(self.data, score_name)
