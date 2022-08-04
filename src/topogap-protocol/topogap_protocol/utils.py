##
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MICROSOFT QUANTUM NON-COMMERCIAL License.
##
import logging
import math
import os
import socket
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Set, Union

import holoviews as hv
import numpy as np
import psutil
import xarray as xr
from dask.distributed import Client, LocalCluster

_LOGGER = logging.getLogger(__name__)


def get_tmp_dir():
    tmp_dir = Path(tempfile.gettempdir())
    if os.name != "nt":  # Not on Windows
        tmp = Path("/tmp/scratch")
        tmp.mkdir(exist_ok=True)
        gb_free = psutil.disk_usage(tmp).free >> 30  # free disk space in GB
        if (
            gb_free > 50
            # On gscratch the `/tmp` dir quickly fills up.
            and socket.gethostname() != "QUANTUM-NFS-SERVER-001"
        ):
            tmp_dir = tmp
    return tmp_dir


TMP_DIR = get_tmp_dir()

def load_phase2_data(filename):
    ds_left = xr.open_dataset(filename, format='NETCDF4', engine='netcdf4', group='left')
    for k in ['g_rr', 'g_lr', 'i_l', 'i_r']:
        ds_left[k] = 0.0 * ds_left['g_ll']
    
    ds_right = xr.open_dataset(filename, format='NETCDF4', engine='netcdf4', group='right')
    for k in ['g_ll', 'g_rl', 'i_l', 'i_r']:
        ds_right[k] = 0.0 * ds_right['g_rr']
    
    return ds_left, ds_right

def _optimal_chunks(da: xr.DataArray, client: Client, ignore: Iterable[str] = ()):
    n, dim = max((n, d) for d, n in zip(da.dims, da.shape) if d not in ignore)
    n_cores = sum(client.ncores().values()) or 1
    chunk_size = max(math.ceil(n / n_cores), 1)
    return {dim: chunk_size}


class DaskClient:
    def __init__(self, threads_per_worker=1):
        self.threads_per_worker = threads_per_worker
        self._client = None

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.client, attr)

    @property
    def client(self):
        if self._client is None:
            print("Starting a `dask.Client` with a `distributed.LocalCluster`.")
            cluster = LocalCluster(
                threads_per_worker=self.threads_per_worker, local_directory=TMP_DIR
            )
            self._client = Client(cluster)
        return self._client


dask_client = DaskClient()


def apply_func_along_dim(
    da: xr.DataArray,
    dim: str,
    func: Callable[..., Any],
    func_kwargs: Optional[Dict[str, Any]] = None,
    parallel: Union[bool, str] = "auto",
):
    # _, max_dim = max((n, d) for d, n in zip(da.dims, da.shape) if d != dim)
    # chunked_da = da.chunk({dim: -1, max_dim: "auto"})
    # Using _optimal_chunks gave some memory issues, we can use the "auto"
    # argument from above, but it's slower.
    if parallel == "auto":
        parallel = os.name != "nt"

    if parallel:
        da = da.chunk(_optimal_chunks(da, dask_client, ignore=[dim]))
    x = xr.apply_ufunc(
        func,
        da[dim],
        da,
        kwargs=func_kwargs or {},
        input_core_dims=[[dim], [dim]],
        vectorize=True,
        dask="parallelized" if parallel else "allowed",
        output_dtypes=[dict],
        keep_attrs=True,
    )
    if not parallel:
        return x
    return dask_client.compute(x, traverse=False, optimize_graph=False, sync=True)


def which_bias(data, name):
    if {"lb", "rb"}.issubset(data.dims):
        # assumes name is in the form "xc_yb"
        # where x=l,r is cutter side and y=l,r is bias side
        bias_side = name[3]
        if bias_side == "l":
            return "lb"
        elif bias_side == "r":
            return "rb"
        else:
            raise NameError(
                "Dataset name was not in the form 'xc_yb'. Could not determine bias side."
            )
    elif "lb" in data.dims:
        return "lb"
    elif "rb" in data.dims:
        return "rb"
    else:
        raise NameError(
            "Neither 'lb' nor 'rb' are dimensions of this Dataset. Cannot proceed."
        )


def _holomap(data, name, vdims):
    ds = data[name]
    hv_ds = hv.Dataset(ds)
    kdims = ["p", which_bias(ds, name)]
    plots = [
        hv_ds.to(hv.Image, kdims=kdims, vdims=[vdim], label=vdim, dynamic=True)
        .options(aspect="square", axiswise=True, framewise=True)
        .hist()
        for vdim in vdims
    ]
    return hv.Layout(plots).cols(2)


def drop_dimensionless_variables(ds, skip: Optional[Set] = None):
    skip = skip or set()
    to_drop = [
        coord.name
        for coord in ds.coords.values()
        if coord.dims == () and coord.name not in skip
    ]
    return ds.drop_vars(to_drop)


def broaden(ds, temp=4.5e-6, bias_name="bias"):
    def dfermi(x, t):
        return np.exp(-x / t) / (t * (1 + np.exp(-x / t)) ** 2)

    def conv(bias, array):
        return (
            np.convolve(array, dfermi(bias, temp), mode="same")
            * (bias[1] - bias[0]).data
        )

    broadened = xr.apply_ufunc(
        conv,
        ds[bias_name],
        ds,
        input_core_dims=[[bias_name], [bias_name]],
        output_core_dims=[[bias_name]],
        vectorize=True,
    )
    return broadened


def is_homogeneous(arr):
    xs = np.linspace(arr[0], arr[-1], len(arr))
    return np.allclose(arr, xs)


