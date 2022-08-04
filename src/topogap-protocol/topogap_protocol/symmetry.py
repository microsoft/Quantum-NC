##
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MICROSOFT QUANTUM NON-COMMERCIAL License.
##
from typing import Union

import numpy as np


def flip(bias: Union[np.ndarray, int], cond: np.ndarray) -> np.ndarray:
    """
    Flips the 1D list conductance around the zero of the bias list
    or taking into account an integer offset.

    Parameters
    ----------
    bias : 1D numpy array or int
        If ``bias`` is a 1D numpy array, the center of the flip is determined by the
        zero in the bias list, otherwise if ``bias`` is a int, it's interpreted as
        an integer offset that is taken into account when flipping ``cond``.
    cond : 1D numpy array
        List of conductance values as function of bias to be flipped.

    Returns
    -------
    flipped_cond : 1D numpy array
        Flipped version of input list.
    """
    if isinstance(bias, (int, np.integer)):
        return np.roll(cond[::-1], 2 * bias)
    else:  # input is bias array
        # determine distance of zero of bias list from center
        zerobias_shift = np.argmin(np.abs(bias)) - len(bias) // 2
        return np.roll(cond[::-1], 2 * zerobias_shift)


def antisymmetric_conductance_part(
    bias: np.ndarray, cond: np.ndarray, shift=True
) -> np.ndarray:
    """Returns the anti-symmetric part of the conductance.

    Parameters
    ----------
    bias : 1D numpy.array
        Bias values.
    cond : 1D or 2D numpy array
        Conductance values.
    shift : Boolean
        whether to shift bias or not

    Returns
    -------
    a_cond : 1D or 2D numpy array
        Depending on shape of the input ``cond``, ``a_cond`` is a
        1D or 2D array of the antisymmetric part of cond.
    """
    # 1D array
    if len(cond.shape) == 1:
        return (cond - flip(bias, cond)) / 2
    # 2D array
    elif len(cond.shape) == 2:
        zerobias_shift = np.argmin(np.abs(bias)) - len(bias) // 2
        # Recursively call self
        if not shift:
            zerobias_shift = 0
        conds = [
            antisymmetric_conductance_part(zerobias_shift, cond_1d)
            for cond_1d in np.transpose(cond)
        ]
        return np.transpose(conds)
    else:
        raise ValueError("``cond`` has be a 1D or 2D array.")
