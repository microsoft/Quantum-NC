##
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MICROSOFT QUANTUM NON-COMMERCIAL License.
##
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist

from topogap_protocol.find_peaks import create_zbp_map, find_peaks_or_dips_in_data_array
from topogap_protocol.specs import (
#     CalibrationData,
    PhaseOneData,
    PhaseTwoRegionData,
#     RFData,
)
from topogap_protocol.utils import drop_dimensionless_variables


def _zbp_extraction_per_side(
    phase_one_or_two_region: Union[PhaseOneData, PhaseTwoRegionData],
    side: str,
    relative_prominence=0.04,
    relative_height=0.04,
    filter_params=dict(window_length=21, polyorder=2),
    peak_params=None,
    bias_threshold=1e-5,
    parallel: Union[bool, str] = "auto",
) -> xr.DataArray:
    """
    Finds ZBPs in either left or right conductance data.
    ZBV values are averaged over cutter to return a
    ZBP frequency (probability) as a function of other parameters

    Parameters
    ----------
    phase_one_or_two_region : `PhaseOneData` or `PhaseTwoRegionData`
        A `PhaseOneData` or `PhaseTwoRegionData` instance.
    side: str
        Either ``'l', 'left', ...`` or ``'r', 'right', ...``
        Determines whether to search left or right data.
    relative_prominence, relative_height, filter_params, peak_params
        See `find_peaks_or_dips_in_data_array`.
    bias_threshold : float
        Maximum bias voltage for which a peak classifies as a zero bias peak
        for sim data set, the value is set to be 5x resolution in bias voltage.
    parallel : Union[bool, Literal["auto"]]
        Call `find_peaks_or_dips_in_trace` in parallel. If "auto" the
        parallelization is enabled on Linux but disabled on Windows.

    Returns
    -------
    xr.DataArray
        A data array of the probability that the dataset
        contains a ZBP.
    """
    bias_name = f"{side[0]}b"
    cutter_name = f"{side[0]}c"

    if isinstance(phase_one_or_two_region, PhaseOneData):
        data = phase_one_or_two_region.data[f"{cutter_name}_{bias_name}"]
    elif isinstance(phase_one_or_two_region, PhaseTwoRegionData):
        data = phase_one_or_two_region.data[side]
    else:
        raise NotImplementedError(
            "Only supports `PhaseOneData` or `PhaseTwoRegionData`."
        )

    g_da = data[f"g_{side[0]}{side[0]}"].load()

    peaks_da = find_peaks_or_dips_in_data_array(
        g_da,
        coord=bias_name,
        relative_prominence=relative_prominence,
        relative_height=relative_height,
        filter_params=filter_params,
        peak_params=peak_params or {},
        dips=False,
        parallel=parallel,
    )
    zbp = create_zbp_map(peaks_da, bias_threshold=bias_threshold)
    return zbp.mean(cutter_name)


def zbp_dataset(
    phase_one_or_two_region: Union[PhaseOneData, PhaseTwoRegionData],
    threshold: float,
    bias_threshold: float = 1e-5,
    relative_prominence=0.04,
    relative_height=0.04,
    filter_params=dict(window_length=21, polyorder=2),
    peak_params=None,
    parallel: Union[bool, str] = "auto",
) -> xr.Dataset:
    """
    Adds to data a new Dataset "zbp" containing
    ZBP maps for left and right sides, and a joint map indicating
    where ZBPs are present with a probability higher than a certain threshold.

    Parameters
    ----------
    phase_one_or_two_region : `PhaseOneData` or `PhaseTwoRegionData`
        A `PhaseOneData` or `PhaseTwoRegionData` instance.
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


    Returns
    -------
    xarray.Dataset
        A dataset containing information about the postions of ZBPs.
    """
    kwargs = dict(
        relative_prominence=relative_prominence,
        relative_height=relative_height,
        filter_params=filter_params,
        peak_params=peak_params,
        parallel=parallel,
        bias_threshold=bias_threshold,
    )
    left = _zbp_extraction_per_side(phase_one_or_two_region, "left", **kwargs).squeeze()
    right = _zbp_extraction_per_side(
        phase_one_or_two_region, "right", **kwargs
    ).squeeze()
    zbp_ds = xr.Dataset(
        {"left": left, "right": right, "zbp": (left > threshold) * (right > threshold)}
    )
    zbp_ds["zbp"].attrs["threshold"] = threshold

    # Set attrs
    zbp_ds.attrs["threshold"] = threshold
    zbp_ds.attrs["bias_threshold"] = bias_threshold
    zbp_ds.attrs["relative_prominence"] = relative_prominence
    zbp_ds.attrs["relative_height"] = relative_height
    zbp_ds.attrs["filter_params"] = filter_params
    zbp_ds.attrs["peak_params"] = peak_params
    return zbp_ds


def zbp_dataset_derivative(
    phase_one_or_two_region: Union[PhaseOneData, PhaseTwoRegionData],
    derivative_threshold: float,
    probability_threshold: float = 1.0,
    bias_window: float = 3e-2,
    polyorder: int = 2,
    average_over_cutter: bool = True,
):
    """
    Adds to data a new Dataset "zbp" containing
    ZBP maps for left and right sides, and a joint map indicating
    where ZBPs are present with a probability higher than a certain threshold.

    Using the `scipy.signal.savgol_filter` to approximate the 2nd derivative.

    Parameters
    ----------
    phase_one_or_two_region : `PhaseOneData` or `PhaseTwoRegionData`
        A `PhaseOneData` or `PhaseTwoRegionData` instance.
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
    kwargs = dict(
        bias_window=bias_window, polyorder=polyorder, average_over_cutter=False
    )
    left = smooth_deriv_per_side(phase_one_or_two_region, side="left", **kwargs)
    right = smooth_deriv_per_side(phase_one_or_two_region, side="right", **kwargs)
    P_left = (left <= -derivative_threshold).squeeze()
    P_right = (right <= -derivative_threshold).squeeze()
    if average_over_cutter:
        P_left = P_left.mean("lc")
        P_right = P_right.mean("rc")
    zbp_ds = xr.Dataset(
        {
            "left": P_left,
            "right": P_right,
            "zbp": (P_right >= probability_threshold)
            * (P_left >= probability_threshold),
        }
    )
    zbp_ds = drop_dimensionless_variables(zbp_ds)
    zbp_ds.attrs["derivative_threshold"] = derivative_threshold
    zbp_ds.attrs["probability_threshold"] = probability_threshold
    zbp_ds.attrs["bias_window"] = bias_window
    zbp_ds.attrs["polyorder"] = polyorder
    zbp_ds.attrs["average_over_cutter"] = average_over_cutter
    return zbp_ds


def smooth_deriv_per_side(
    phase_one_or_two_region: Union[PhaseOneData, PhaseTwoRegionData],
    side: str = "left",
    average_over_cutter: bool = True,
    bias_window: float = 3e-2,
    deriv: int = 2,
    polyorder: int = 2,
):
    bias_name = f"{side[0]}b"
    cutter_name = f"{side[0]}c"
    other_side = {"left": "right", "right": "left"}[side]

    if isinstance(phase_one_or_two_region, PhaseOneData):
        data = phase_one_or_two_region.data[f"{cutter_name}_{bias_name}"]
    elif isinstance(phase_one_or_two_region, PhaseTwoRegionData):
        data = phase_one_or_two_region.data[side]
    else:
        raise NotImplementedError(
            "Only supports `PhaseOneData` or `PhaseTwoRegionData`."
        )

    g_da = data[f"g_{side[0]}{side[0]}"].load()
    bias_name = f"{side[0]}b"

    # Calculate window_length using the bias axis
    coords = list(g_da.coords)
    axis = coords.index(bias_name)
    bias = data[bias_name]
    delta = np.mean(np.diff(bias))
    if np.max(np.abs(bias)) > bias_window:
        window_length = int(2 * bias_window / delta)
    else:
        raise ValueError(
            "Bias window must be smaller or equal than the bias range in the data"
        )
    if window_length % 2 == 0:
        window_length += 1

    g_smooth = xr.apply_ufunc(
        savgol_filter,
        g_da,
        window_length,
        polyorder,
        deriv,
        delta,
        axis,
        input_core_dims=[coords, [], [], [], [], []],
        output_core_dims=[coords],
    )
    g_2nd_deriv = g_smooth.sel({bias_name: 0.0}, method="nearest", drop=True)
    g_2nd_deriv = g_2nd_deriv.squeeze(f"{other_side[0]}b", drop=True)
    if average_over_cutter:
        return g_2nd_deriv.mean(cutter_name).mean(f"{other_side[0]}c")
    else:
        return g_2nd_deriv


# def distance_from_calibration_data(
#     phase_one: PhaseOneData,
#     calibration_data: CalibrationData,
#     rf_data: RFData,
#     name: str,
#     side: str,
#     f_idx: int,
#     *,
#     lock: threading.Lock,
# ) -> float:
#     """
#     Comparison of rf data and calibration data taken at a certain field.
#     The distance between the two dataset is given by the average absolute value of the
#     difference between the estimated conductance of each rf data point, and the conductance measured
#     at the calibration data point which is closest to it on the complex plane.
#     """
#     # RF and conductance data as list of coords with shape (N, 2)
#     f = rf_data.f_values[f_idx]
#     with lock:  # .sel is not threadsafe
#         rf_ds = rf_data.data[name].sel(f=f, method="nearest")
#     re = rf_ds[f"rf_{side[0]}_re"].values
#     im = rf_ds[f"rf_{side[0]}_im"].values
#     z = np.stack([re, im], axis=-1)
#     z = z.reshape(-1, 2)
#     with lock:
#         g_ds = phase_one.data[name].sel(f=f, method="nearest")
#     g = g_ds[f"g_{side[0]}{side[0]}"].values.flatten()
# 
#     # Calibration data as list of coords with shape (N, 2)
#     with lock:
#         cal_ds = calibration_data.data[side].sel(f=f, method="nearest")
#     w = np.stack([cal_ds["re"].values, cal_ds["im"].values], axis=-1)
#     gcal = cal_ds["g"].values
# 
#     # For each rf point, get conductance of closest calibration point
#     dist = cdist(z, w)
#     gmin = gcal[np.argmin(dist, axis=1)]
#     gdist = np.average(np.abs(g - gmin))
#     return gdist


# def calibration_check(
#     phase_one: PhaseOneData,
#     calibration_data: CalibrationData,
#     rf_data: RFData,
# ) -> pd.DataFrame:
#     lock = threading.Lock()
# 
#     def _dist(name_side_i, lock=lock):
#         name, side, i = name_side_i
#         return distance_from_calibration_data(
#             phase_one=phase_one,
#             calibration_data=calibration_data,
#             rf_data=rf_data,
#             name=name,
#             f_idx=i,
#             side=side,
#             lock=lock,
#         )
# 
#     fs = rf_data.f_values
#     with ThreadPoolExecutor() as ex:
#         var_names = rf_data.get_standard_xarray_names()
#         if rf_data.cross_sweeps:
#             combos = [
#                 (name, side, i)
#                 for name in var_names
#                 for side in ("left", "right")
#                 for i, _ in enumerate(fs)
#             ]
#         else:
#             side_map = {"l": "left", "r": "right"}
#             combos = [
#                 (name, side_map[name[0]], i)
#                 for name in var_names
#                 for i, _ in enumerate(fs)
#             ]
#         distances = list(ex.map(_dist, combos))
# 
#     cols = ["name", "side", "i"]
#     df_index = pd.DataFrame(combos, columns=cols)
#     df_dists = pd.DataFrame(distances, columns=["distance"])
#     df_f = pd.DataFrame(list(fs) * (len(combos) // len(fs)), columns=["f"])
#     df = pd.concat((df_dists, df_f, df_index), axis=1)
#     df = df.set_index(cols)
#     return df


def flatten_clusters(
    ds: xr.Dataset,
    plunger_name: str = "p",
    field_name: str = "f",
    cluster_name: str = "zbp_cluster",
    zbp_cluster_number_name: str = "zbp_cluster_number",
) -> xr.DataArray:

    return xr.apply_ufunc(
        lambda x, i_cluster: (x * i_cluster[:, None, None]).sum(axis=0),
        ds[cluster_name],
        np.arange(1, len(ds[zbp_cluster_number_name]) + 1),
        input_core_dims=[
            [zbp_cluster_number_name, field_name, plunger_name],
            [zbp_cluster_number_name],
        ],
        output_core_dims=[[field_name, plunger_name]],
        vectorize=True,
    )
