##
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MICROSOFT QUANTUM NON-COMMERCIAL License.
##
"""A module for analyzing conductance matrix symmetry relations and gaps."""

from typing import Tuple, Union

import numpy as np
import scipy.ndimage
import xarray as xr

from topogap_protocol.symmetry import antisymmetric_conductance_part


def extract_gap_from_trace(bool_array: np.ndarray, bias_array: np.ndarray) -> float:
    """Extracts the gap values from a 1D binary array in units of the given bias array.

    Parameters
    ----------
    bool_array : np.ndarray
        Boolean 1D array.
    bias_array : np.ndarray
        1D array of bias values.

    Returns
    -------
    float
        gap
    """
    i_zero = np.argmin(np.abs(bias_array))  # Index of the bias zero.
    gap = bias_array[i_zero:][bool_array[i_zero:]]
    if gap.size == 0:
        return bias_array[-1]
    else:
        return gap[0]


def determine_gap_2D(
    data_2D: xr.Dataset,
    bias_name: str,
    field_name: str = "f",
    median_size: float = 2,
    gauss_sigma: Tuple[float, float] = (1, 2),
    threshold_factor: float = 0.1,
    filtered_antisym_g: bool = False,
    shift=True,
) -> Union[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    """
    Extracts the gap from a 2D array (non-local conductance vs bias and field)
    using thresholding of filtered data.

    Parameters
    ----------
    data_2D : xarray.DataArray
        (non-local conductance vs bias, field)
    bias_name : str
        Name of bias axis.
    field_name : str
        Name of field axis.
    median_size : float, optional
        Size of the median filter, ``size`` argument to `scipy.ndimage.median_filter`.
    gauss_sigma : float, optional
        Size of the gaussian filter, ``sigma`` argument to `scipy.ndimage.gaussian_filter`.
    threshold_factor : float, optional
        Threshold is `threshold_factor * max(array)`.
    filtered_antisym_g : bool, optional
        Include the filtered_antisym_g from which the gap
        is extracted.
    shift : bool, optional
        Whether to shift bias for antisymmetric part of conductance

    Returns
    -------
    xarray.DataArray or Tuple[xarray.DataArray, xarray.DataArray]
        Estimates of the gap.
    """
    data_2D = data_2D.load()
    defaults = dict(
        input_core_dims=[[bias_name, field_name]],
        output_core_dims=[[bias_name, field_name]],
        vectorize=True,
    )
    conductance = xr.apply_ufunc(
        scipy.ndimage.median_filter, data_2D, kwargs=dict(size=median_size), **defaults
    )
    antisymmetric_conductance = xr.apply_ufunc(
        antisymmetric_conductance_part,
        data_2D[bias_name],
        conductance,
        input_core_dims=[[bias_name], [bias_name, field_name]],
        output_core_dims=[[bias_name, field_name]],
        kwargs={"shift": shift},
        vectorize=True,
    )
    filtered = xr.apply_ufunc(
        scipy.ndimage.gaussian_filter,
        antisymmetric_conductance,
        kwargs=dict(sigma=gauss_sigma),
        **defaults,
    )

    def _thresholding_max(x):
        return np.abs(x) > threshold_factor * np.max(x)

    binary = xr.apply_ufunc(_thresholding_max, filtered, **defaults)
    gap = xr.apply_ufunc(
        extract_gap_from_trace,
        binary,
        binary[bias_name],
        input_core_dims=[[bias_name], [bias_name]],
        output_core_dims=[[]],
        vectorize=True,
    )

    gap = gap.rename("gap")
    return (gap, filtered) if filtered_antisym_g else gap
