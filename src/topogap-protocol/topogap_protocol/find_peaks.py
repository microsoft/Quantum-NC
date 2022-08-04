##
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MICROSOFT QUANTUM NON-COMMERCIAL License.
##
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import xarray as xr
from ipywidgets import interactive

from topogap_protocol.utils import apply_func_along_dim


def find_peaks_or_dips_in_trace(
    x: np.ndarray,
    y: np.ndarray,
    relative_prominence: float,
    relative_height: float,
    peak_params: Dict[str, Any],
    filter_params: Dict[str, int] = None,
    dips: bool = False,
    *,
    full_output: bool = False,
):
    """Find peaks in a 1D trace.

    Parameters
    ----------
    x : np.ndarray
        Array of values on the x-axis.
    y : np.ndarray
        Array of values on the y-axis.
    relative_prominence : float between 0 and 1
        Relative threshold for the prominence of peaks.
    relative_height : float between 0 and 1
        Relative threshold for peak heights.
    peak_params : dict
        Dictionary of parameters to be passed to find_peaks.
    filter_params : dict
        Dictionary of parameters to be passed to savgol_filter:
        ``dict(window_length=..., polyorder=...)``.
    dips : bool
        If True, looks for dips instead of peaks.
    full_output : bool, optional
        Return the x and y (+filtered) values, by default False.

    Returns
    -------
    dict
        Dictionary with peak information.
    """
    if np.isnan(y).any():
        # If the 1D trace contains NaN, we ignore it and go on,
        # scipy.signal.find_peaks is ill behaved with NaNs.
        return np.nan

    out = {}
    if filter_params is not None:
        yf = scipy.signal.savgol_filter(y, **filter_params)  # smoothen raw data
    else:
        yf = y
    prominence = np.ptp(yf) * relative_prominence
    y_dip = -yf if dips else yf
    height = np.ptp(yf) * relative_height + np.min(y_dip)
    out["i_peaks"], out["properties"] = scipy.signal.find_peaks(
        y_dip, height=height, prominence=prominence, **peak_params
    )
    out["x_peaks"] = x[out["i_peaks"]]
    out["y_peaks"] = yf[out["i_peaks"]]
    out["n_peaks"] = len(out["i_peaks"])

    if full_output:
        out["x"] = x
        out["y"] = y
        if filter_params is not None:
            out["y_filtered"] = yf
    return out


def find_peaks_or_dips_in_data_array(
    array: xr.DataArray,
    coord: str,
    relative_prominence: float,
    relative_height: float,
    filter_params: Dict[str, int],
    peak_params: Dict[str, Any],
    dips: bool = False,
    parallel: Union[bool, str] = "auto",
) -> xr.DataArray:
    """
    Peak finder for a data array with arbitrary number of dimensions.

    It runs scipy.signal.find_peaks for all the 1D traces in the data along
    a given coordinate (iterating over all other coordinates).

    The function smoothens the data first, using Savitzky-Golay
    filter (scipy.signal.savgol_filter) if ``filter_params`` is not None.

    Peaks are filtered according to their prominence and height.
    Relative thresholds for both are passed as parameters.
    Absolute values for the thresholds are determined trace by trace,
    using the relative ones times the total range of the trace (`numpy.ptp`).
    Additional thresholds (e.g. ``width``) can be passed using ``peak_params``.
    See documentation of `scipy.signal.find_peaks`.

    Parameters
    ----------
    array : xarray.DataArray
        Input data.
    coord : string, e.g. 'Bias'
        The coordinate along which to find peaks.
    relative_prominence : float between 0 and 1
        Relative threshold for the prominence of peaks.
    relative_height : float between 0 and 1
        Relative threshold for peak heights.
    filter_params : dict
        Dictionary of parameters to be passed to savgol_filter:
        ``dict(window_length=..., polyorder=...)``.
    peak_params : dict
        Dictionary of parameters to be passed to find_peaks.
    dips : bool
        If True, looks for dips instead of peaks.
    parallel : Union[bool, Literal["auto"]]
        Call `find_peaks_or_dips_in_trace` in parallel. If "auto" the
        parallelization is enabled on Linux but disabled on Windows.

    Returns
    -------
    peaks_da: xarray.DataArray
        An DataArray with the shape of array expect with
        the dimension ``coord`` squashed.
        The values are dictionaries with the return values
        of `find_peaks_or_dips_in_trace`.
    """
    peaks_da = apply_func_along_dim(
        array,
        dim=coord,
        func=find_peaks_or_dips_in_trace,
        func_kwargs=dict(
            relative_prominence=relative_prominence,
            relative_height=relative_height,
            peak_params=peak_params,
            filter_params=filter_params,
            dips=dips,
        ),
        parallel=parallel,
    )
    return peaks_da


# XXX: add to specs.py
def interactive_peak_finder(
    data: xr.Dataset,
    coord: str,
    variable: str,
    bias_threshold: float,
    filter_params: Dict[str, int] = dict(window_length=21, polyorder=2),
    peak_params: Dict[str, Any] = {},
    dips: bool = False,
) -> interactive:
    """Interactively scroll through traces and tweak peak finding parameters.

    Parameters
    ----------
    data : xr.Dataset
        Dataset with conductance.
    coord : str
        Name of the x-axis values.
    variable : str
        Name of the y-axis values.
    bias_threshold : float
        Range in which a peak is considered a ZBP.
    filter_params : Dict[str, int], optional
        Dictionary of parameters to be passed to savgol_filter
        by default dict(window_length=21, polyorder=2).
    peak_params : Dict[str, Any], optional
        Dictionary of parameters to be passed to find_peaks, by default {}.
    dips : bool, optional
        If True, looks for dips instead of peaks, by default False.

    Returns
    -------
    ipywidgets.interactive
        An interacive widget to find peaks.
    """

    def show_linecut_with_peaks(relative_prominence, relative_height, **other_coords):
        xvalues = data[coord].values
        yvalues = data[other_coords][variable].values

        peak_info = find_peaks_or_dips_in_trace(
            xvalues,
            yvalues,
            relative_prominence,
            relative_height,
            peak_params,
            filter_params,
            dips,
            full_output=True,
        )

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        if filter_params:
            ax.plot(xvalues, yvalues, ".")
            ax.plot(xvalues, peak_info["y_filtered"], c="k")
        else:
            ax.plot(xvalues, yvalues, c="k")

        for x_peak in peak_info["x_peaks"]:
            ax.axvline(x_peak, c="k", lw=0.5)

        # Add a red band around bias_threshold
        ax.axvspan(-bias_threshold, bias_threshold, alpha=0.2, color="red")

        ax.set_title(
            f"{peak_info['n_peaks']} peaks were found."
            f" Relative prominence = {relative_prominence:.3f}."
            f" Relative height = {relative_height:.3f}"
        )
        ax.set_xlabel(coord)
        ax.set_ylabel(variable)
        fig.show()

    dims = {dim: (0, data.sizes[dim] - 1) for dim in data.dims if dim != coord}
    return interactive(
        show_linecut_with_peaks,
        relative_prominence=(0, 1, 0.001),
        relative_height=(0, 1, 0.001),
        **dims,
    )


def create_zbp_map(peaks_da: xr.Dataset, bias_threshold: float) -> xr.Dataset:
    """
    Builds a map with the position of ZBPs in phase space.

    Parameters
    ----------
    peaks_da : xarray.Dataset
        Output DataArray of `find_peaks_or_dips_in_data_array`.
    bias_threshold : float
        Maximum bias voltage for which a peak classifies as a zero bias peak
        for sim data set, the value is set to be 5x resolution in bias voltage.

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing the positions of all the zero bias peaks.
        the coordinates of the array coincide with those of the reduced data (except for 'Bias')
        the dataset has one boolean data variable 'ZBP'.
    """

    def condition(peak_info):
        if isinstance(peak_info, float) and np.isnan(peak_info):
            return False
        return any(abs(x) < bias_threshold for x in peak_info["x_peaks"])

    zbp = xr.apply_ufunc(condition, peaks_da, vectorize=True)
    zbp.name = "ZPB"
    return zbp
