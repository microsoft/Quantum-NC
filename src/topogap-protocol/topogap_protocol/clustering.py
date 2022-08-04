##
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MICROSOFT QUANTUM NON-COMMERCIAL License.
##
from contextlib import suppress
from functools import partial
from typing import Any, Dict, List, Set, Tuple

import jenkspy
import numpy as np
import scipy.linalg as la
import sklearn
import sklearn.cluster
import xarray as xr
from scipy import ndimage

from topogap_protocol.utils import drop_dimensionless_variables


def get_boundary_indices(
    cluster_array: np.ndarray,
    threshold_clustering: float = 1.05,
    silhouette_threshold: float = 0.63,
    method: str = "old",
) -> Set[Tuple[int, int]]:
    """Returns indices of the points at the boundary of a cluster.

    Parameters
    ----------
    cluster_array : np.ndarray
        A boolean array of a cluster.
    threshold_clustering : float, optional
        Allowed decrease in standard deviation between ``n`` and ``n+1``
        clusters when the extra cluster is considered not necessary. Used
        if Jenks breaks algorithm is used, by default 1.05.
    silhouette_threshold : float, optional
        Threshold below which the clustering is assumed to be bad. Used if
        KMeans algorithm is used, by default 0.63.
    method : str, optional
        Uses ``sklearn.clustering.KMeans`` for clustering unless `method="old"`,
        then it uses the Jenks breaks, by default "old".
    Returns
    -------
    Set[Tuple[int, int]]
    """
    if method == "old":
        f_cluster = partial(
            find_optimal_clustering_1D_own,
            threshold_clustering=threshold_clustering,
        )
    else:
        f_cluster = partial(
            find_optimal_clustering_1D_sklearn,
            silhouette_threshold=silhouette_threshold,
        )
    nonzero = cluster_array != 0
    boundary = set()  # set with (i, j) tuples of the boundary
    for i, row in enumerate(nonzero):
        if not row.any():
            continue
        clusters = f_cluster(row)
        for cluster in clusters:
            indices = np.where(cluster)[0]
            start, end = indices[0], indices[-1]
            if start > 0:
                boundary.add((i, start))
            if end < (len(cluster) - 1):
                boundary.add((i, end))

    for j, col in enumerate(nonzero.T):
        if not col.any():
            continue
        clusters = f_cluster(col)
        for cluster in clusters:
            indices = np.where(cluster)[0]
            start, end = indices[0], indices[-1]
            if start > 0:
                boundary.add((start, j))
            if end < (len(cluster) - 1):
                boundary.add((end, j))
    return boundary


def gapless_boundary(
    boundary_indices: Set[Tuple[int, int]],
    gap_closed_array: np.ndarray,
    variance: int = 5,
):
    """Returns the percentage of the boundary of the 2D cluster that is gapless.

    Parameters
    ----------
    boundary_indices: Set[Tuple[int, int]]
        Set of indices of the points at the boundary of a cluster.
    gap_closed_array : np.ndarray
        Boolean array variables with True signifying closed gap.
    variance : int, optional
        Position tolerance for distance between the ZBP array end and the gap
        closing, by default 5.

    Returns
    -------
    Tuple[float, bool, Dict[Tuple[int, int], bool]]]
        - Percentage of the boundary that is gapless.
        - Dictionary with boundary indices -> bool
    """
    s = max(1, variance // 2)
    n, m = gap_closed_array.shape
    topo_boundary = {}
    for i, j in boundary_indices:
        gap_is_close = gap_closed_array[
            max(0, i - s) : min(i + s + 1, n),
            max(0, j - s) : min(j + s + 1, m),
        ].any()
        topo_boundary[i, j] = gap_is_close
    percentage = (
        sum(topo_boundary.values()) / len(topo_boundary) if topo_boundary else 0
    )
    return percentage, topo_boundary


def find_optimal_clustering_1D_sklearn(
    zbp_array: np.ndarray, silhouette_threshold: float = 0.63
) -> List[np.ndarray]:
    """Function to find optimal clustering in 1D using ``sklearn.cluster.KMeans``.

    Finds the optimal number of clusters using silhouette score.

    Parameters
    ----------
    zbp_array : np.ndarray
        Binary array of data with 1 representing present ZBP, 0 -- absent.
    silhouette_threshold : float, optional
        Threshold below which the clustering is assumed to be bad, by default 0.63.

    Returns
    -------
    List[np.ndarray]
        List of boolean arrays corresponding to only points in each cluster.
    """
    if la.norm(zbp_array) == 0:
        return []
    to_cluster = np.nonzero(zbp_array)[0]
    for i in range(2, np.size(to_cluster)):
        model = sklearn.cluster.KMeans(n_clusters=i).fit(to_cluster.reshape(-1, 1))
        labels = model.labels_
        score = sklearn.metrics.silhouette_score(
            to_cluster.reshape(-1, 1), labels, metric="euclidean"
        )
        if score < silhouette_threshold:
            clusters = []
            positions = sklearn.cluster.KMeans(n_clusters=i - 1).fit_predict(
                to_cluster.reshape(-1, 1)
            )
            for j in range(i - 1):
                zbp_array0 = np.zeros(np.size(zbp_array), dtype=bool)
                zbp_array0[to_cluster[positions == j]] = True
                clusters.append(zbp_array0)
            return clusters
    return []


def gap_statistics(
    cluster_array: np.ndarray, gap_array: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    """Given a cluster array and array of gap values (computed or deduced),
    this function returns average, maximal, and minimal (sign-resolved) gap.

    Parameters
    ----------
    cluster_array : np.ndarray
        A boolean array of a cluster.
    gap_array : np.ndarray
        Array of gap values for all data points.

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        - minimal gap
        - maximal gap
        - average gap (with sign)
        - median gap (with sign)
        - average absolute gap (without sign)
        - median absolute gap (without sign)
    """
    gaps = gap_array[np.nonzero(cluster_array)]
    if gaps.size == 0:
        return 6 * (np.nan,)
    abs_gap = np.abs(gaps)
    return (
        gaps.min(),
        gaps.max(),
        gaps.mean(),
        np.median(gaps),
        abs_gap.mean(),
        np.median(abs_gap),
    )


def extract_clusters_local_2D(
    data_2D: xr.Dataset,
    name_zbp: str = "zbp",
    name_cluster: str = "zbp_cluster",
    name_cluster_number: str = "zbp_cluster_number",
    min_samples: int = 2,
    xi: float = 0.1,
    min_cluster_size: int = 0.01,
    max_eps: float = 10.0,
    **cluster_kwargs: Dict[str, Any],
) -> xr.Dataset:
    """Adds boolean data on the positions of the clusters deduced from the data to ``data_2D``.

    Parameters
    ----------
    data_2D : xarray.Dataset
        Dataset of plunger vs field.
    name_zbp : str, optional
        Name of boolean DataArray, by default "zbp".
    name_cluster : str, optional
        Name for the new column corresponding to the boolean cluster positions, by default "zbp_cluster".
    name_cluster_number : str, optional
        Name for the new coordinate labeling clusters in the data, by default "zbp_cluster_number".
    min_samples : int, optional
        Parameter for `sklearn.cluster.OPTICS`, by default 2.
    xi : float, optional
        Parameter for `sklearn.cluster.OPTICS`, by default 0.1.
    min_cluster_size : int, optional
        Parameter for `sklearn.cluster.OPTICS`, by default 0.01
    max_eps : float, optional
        Parameter for `sklearn.cluster.OPTICS`, by default 10.
    cluster_kwargs : dict
        Keyword arguments passed to the `sklearn.cluster.OPTICS` function.

    Returns
    -------
    xarray.Dataset
    """
    with suppress(KeyError):
        # Remove older instances
        del data_2D[name_cluster_number]
        del data_2D[name_cluster]
        to_drop = [
            k for k, v in data_2D.data_vars.items() if name_cluster_number in v.dims
        ]
        data_2D = data_2D.drop_vars(to_drop)

    def _empty_da(i, da=data_2D[name_zbp]):
        da = drop_dimensionless_variables(xr.zeros_like(da, dtype=int))
        return da.assign_coords({name_cluster_number: i})

    X = np.transpose(np.nonzero(data_2D[name_zbp].data))
    if np.size(X) == 0:
        data_2D[name_cluster] = _empty_da(0).expand_dims(name_cluster_number)
        return data_2D

    if X.shape[0] > min_cluster_size:
        model = sklearn.cluster.OPTICS(
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size,
            max_eps=max_eps,
            **cluster_kwargs,
        )
        labels = model.fit_predict(X)
    else:
        labels = [1] * X.shape[0]

    data_clusters = []
    for i, label in enumerate(np.unique(labels)):
        data_cluster = _empty_da(label)
        for ii, jj in X[labels == label]:
            data_cluster[ii, jj] = 1
        data_clusters.append(data_cluster)

    data_2D[name_cluster] = xr.concat(data_clusters, dim=name_cluster_number)
    return data_2D


def score_nonlocal_2D(
    data_2D: xr.Dataset,
    field_name: str = "f",
    plunger_gate_name: str = "p",
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
    data_2D : xarray.Dataset
        Dataset of plunger vs field.
    field_name : str, optional
        Name of field in ``data_2D``, by default "f".
    plunger_gate_name : str, optional
        Name of plunger_gate_name in ``data_2D``, by default "p".
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
    boundary_indices = xr.apply_ufunc(
        get_boundary_indices,
        data_2D[name_cluster],
        input_core_dims=[[plunger_gate_name, field_name]],
        output_core_dims=[[]],
        vectorize=True,
    )

    percentage_boundary, boundary_indices = xr.apply_ufunc(
        gapless_boundary,
        boundary_indices,
        ~data_2D[name_gap],
        input_core_dims=[[], [plunger_gate_name, field_name]],
        output_core_dims=[[], []],
        vectorize=True,
        kwargs=dict(variance=variance),
    )

    max_gap, average_gap, median_gap = xr.apply_ufunc(
        gap_statistics,
        data_2D[name_cluster],
        data_2D[name_gap_value],
        input_core_dims=[
            [plunger_gate_name, field_name],
            [plunger_gate_name, field_name],
        ],
        output_core_dims=[[], [], [], [], [], []],
        vectorize=True,
    )[1:4]

    data_2D[name_percentage_boundary] = percentage_boundary
    data_2D[name_median_gap] = median_gap
    data_2D[name_average_gap] = average_gap
    data_2D[name_boundary_indices] = boundary_indices
    data_2D[name_median_score] = median_gap * percentage_boundary
    data_2D[name_average_score] = average_gap * percentage_boundary
    data_2D[name_max_score] = max_gap


# XXX: everything below can potentially be removed!


def improve_clustering_1d(n, to_cluster, threshold_clustering=1.05):
    # XXX: is replaced by `find_optimal_clustering_1D_sklearn`?
    """Checks if going from n to n+1 clusters improves the standard deviation
    significantly (by more than threshold_clustering factor)

    Args:
        n: number of clusters
        to_cluster: positions of points in the array to cluster
        threshold_clustering: factor by which the clustering should improve

    Returns:
        result: boolean variable, True if one should go from n to n+1 clusters

    """
    if n > 1:
        breaks_n = jenkspy.jenks_breaks(to_cluster, nb_class=n)
    else:
        breaks_n = [to_cluster[0], to_cluster[-1]]
    if n >= np.size(to_cluster) - 1:
        return False
    breaks_np1 = jenkspy.jenks_breaks(to_cluster, nb_class=n + 1)
    clusters_n = np.std(to_cluster[to_cluster <= breaks_n[1]]) ** 2
    clusters_np1 = np.std(to_cluster[to_cluster <= breaks_np1[1]]) ** 2
    for i in range(n - 1):
        clusters_n += (
            np.std(
                to_cluster[
                    (breaks_n[i + 1] < to_cluster) * (to_cluster <= breaks_n[i + 2])
                ]
            )
            ** 2
        )
    for i in range(n):
        clusters_np1 += (
            np.std(
                to_cluster[
                    (breaks_np1[i + 1] < to_cluster) * (to_cluster <= breaks_np1[i + 2])
                ]
            )
            ** 2
        )
    return clusters_np1 * 2 * threshold_clustering < clusters_n


def find_optimal_clustering_1D_own(
    zbp_array, threshold_clustering=1.05
) -> List[np.ndarray]:
    # XXX: is replaced by `find_optimal_clustering_1D_sklearn`?
    """Function to find optimal clustering in 1D using separation into clusters using jenks_breaks.
    Finds the optimal number of clusters using variance metrics.

    Args:
        zbp_array: binary array of data with 1 representing present ZBP, 0 -- absent
        threshold_clustering: allowed decrease in standard deviation between n and
            n+1 clusters when the extra cluster is considered not necessary

    Returns:
        clusters: List of boolean arrays corresponding to only points in each cluster
    """
    try:
        min_zbp = np.min(np.nonzero(zbp_array))
    except Exception:  # XXX: catch the right error!
        min_zbp = -1

    if min_zbp != -1:
        to_cluster = np.nonzero(zbp_array)[0]
        n = 1
        while improve_clustering_1d(
            n, to_cluster, threshold_clustering=threshold_clustering
        ):
            n += 1
        n_cluster = n
    else:
        n_cluster = 0
    result = []
    if n_cluster != 0:
        if n_cluster > 1:
            breaks = jenkspy.jenks_breaks(to_cluster, nb_class=n_cluster)
        else:
            breaks = [1, np.size(zbp_array) - 1]
        zbp_array0 = np.zeros(np.size(zbp_array))
        zbp_array0[to_cluster[to_cluster <= breaks[1]]] += 1
        result = [zbp_array0]
        for i in range(n_cluster - 1):
            zbp_array0 = np.zeros_like(zbp_array, dtype=bool)
            zbp_array0[
                to_cluster[(breaks[i + 1] < to_cluster) * (to_cluster <= breaks[i + 2])]
            ] = True
            result.append(zbp_array0)
    return result


def cluster_info(
    cluster, plunger_gate_name="p", field_name="f", pct_box=5
) -> Dict[str, Any]:
    p = cluster.mean(field_name)
    p = p[p > 0].coords[plunger_gate_name]
    p_left, p_right = p.min(), p.max()

    f = cluster.mean(plunger_gate_name)
    f = f[f > 0].coords[field_name]
    f_left, f_right = f.min(), f.max()

    dy = np.abs(p_right - p_left) * pct_box / 100
    dx = np.abs(f_right - f_left) * pct_box / 100
    bounding_box = (f_left - dx, p_left - dy, f_right + dx, p_right + dy)

    center = ndimage.measurements.center_of_mass(cluster.data)

    mid_inds = dict(zip(cluster.dims, center))
    mid_inds = {k: round(v) for k, v in mid_inds.items()}

    mid = cluster.isel(**mid_inds)
    center = (mid.coords[field_name].data, mid.coords[plunger_gate_name].data)

    area = cluster.sum().data / np.prod(cluster.data.shape)
    return {"bounding_box": bounding_box, "center": center, "area": area}
