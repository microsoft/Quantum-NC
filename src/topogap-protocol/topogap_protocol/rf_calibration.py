##
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MICROSOFT QUANTUM NON-COMMERCIAL License.
##
import warnings
from copy import copy
from typing import Any, Callable, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import toolz.itertoolz
import xarray as xr
from nptyping import NDArray
from numpy.random import choice
from scipy.optimize import leastsq, minimize


class RFCalibration:
    """A calibration of RF signal vs. DC conductance.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset containing DC conductance and RF signal.
    x_name : str
        Name of RF in-phase component variable.
    y_name : str
        Name of RF out-of-phase component variable.
    g_name : str
        Name of conductance variable.
    verbose : bool
        Prints a report.
    r_sq_threshold : float
        Minimum R² value, see ``should_raise``.
    should_raise : bool
        If True, raises a `RuntimeError` if ``r_sq_threshold`` condition is not met,
        otherwise the values are replaced with NaN.
    min_conductance : float, optional
        Cut off conductance values below this number.
    max_conductance : float, optional
        Cut off conductance values above this number.
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        x_name: str = "re",
        y_name: str = "im",
        g_name: str = "g",
        verbose: bool = False,
        r_sq_threshold: float = 0.95,
        should_raise: bool = True,
        min_conductance: Optional[float] = None,
        max_conductance: Optional[float] = None,
        outlier_cutoff: float = 1.5,
        do_bootstrap: bool = True,
    ):
        self.r_sq_threshold = r_sq_threshold
        self.should_raise = should_raise
        self.min_conductance = min_conductance
        self.max_conductance = max_conductance
        self.do_bootstrap = do_bootstrap
        self.x_name = x_name
        self.y_name = y_name
        self.g_name = g_name
        self.dataset = self.filter(dataset)
        self.fit()
        self.crop_outliers(outlier_cutoff)
        self.fit()

        if verbose:
            self.fit_info()

    def crop_outliers(self, outlier_cutoff: float):
        cal_dataset = self.calibrated_dataset()
        g_model = cal_dataset[self.g_name]
        g = self.g
        residuals = g_model - g
        outlier_masks = []
        for residuals_part in [residuals.real, residuals.imag]:
            residuals_part = residuals.real
            q3 = np.quantile(residuals_part, 0.75)
            q1 = np.quantile(residuals_part, 0.25)
            iqr = q3 - q1
            upper_bound = q3 + outlier_cutoff * iqr
            lower_bound = q1 - outlier_cutoff * iqr
            outliers = (residuals_part > upper_bound) | (residuals_part < lower_bound)
            outlier_masks.append(outliers)
        outliers = outlier_masks[0] | outlier_masks[1]
        self.dataset = self.dataset.where(~outliers, drop=True)

    @property
    def coeffs_as_dict(self):
        coeffs = pack_complex_coeffs(self.coeffs)
        coeffs_dict = dict(zip(["s0", "a", "b"], coeffs))
        coeffs_dict["r_sq"] = self.r_sq
        return coeffs_dict

    def filter(self, dataset: xr.Dataset) -> xr.Dataset:
        """Remove outliers."""
        g = dataset[self.g_name]
        if self.min_conductance is not None:
            dataset = dataset.where(g >= self.min_conductance, drop=True)
        if self.max_conductance is not None:
            dataset = dataset.where(g <= self.max_conductance, drop=True)
        return dataset

    def print_report(self) -> None:
        """Print results of the fit to the calibration dataset."""
        coeffs = pack_complex_coeffs(self.coeffs)
        coeffs_bottom = pack_complex_coeffs(self.coeffs_bottom_q)
        coeffs_top = pack_complex_coeffs(self.coeffs_top_q)
        labels = ["s0", "a", "b"]
        r_sq = goodness_of_fit(self.g, self.s, *coeffs)
        title = "Fit report"
        print(title)
        print("=" * len(title))
        for mean, bottom, top, label in zip(coeffs, coeffs_bottom, coeffs_top, labels):
            msg = f"{label}: {mean:.3g}, 95% confidence rectangle: [{bottom:.3g}, {top:.3g}]"
            print(msg)
        print(f"R²: {r_sq:.3g}")

    @property
    def g(self) -> NDArray[float]:
        """DC conductance."""
        return np.ravel(self.dataset[self.g_name])

    @property
    def s(self) -> NDArray[complex]:
        """RF complex signal."""
        return np.ravel(self.dataset[self.x_name] + 1.0j * self.dataset[self.y_name])

    def fit(self) -> None:
        """Fit a Moebius transformation to the conductance and RF signal."""
        data = np.vstack([self.g, self.s]).T
        if self.do_bootstrap:
            self.coeffs, self.coeffs_bottom_q, self.coeffs_top_q = bootstrap(
                lambda x: unpack_complex_coeffs(*fit_nonlinear(*x.T)), data
            )
        else:
            self.coeffs = unpack_complex_coeffs(*fit_nonlinear(*data.T))
            self.coeffs_bottom_q = [np.nan for _ in self.coeffs]
            self.coeffs_top_q = [np.nan for _ in self.coeffs]
        s0, a, b = pack_complex_coeffs(self.coeffs)
        self.r_sq = goodness_of_fit(self.g, self.s, s0, a, b)
        if self.r_sq < self.r_sq_threshold:
            if self.should_raise:
                raise RuntimeError(
                    f"No good fit found for calibration (R^2={self.r_sq:.3g})."
                )
            else:
                self.coeffs = len(self.coeffs) * [np.nan]
                self.coeffs_bottom_q = copy(self.coeffs)
                self.coeffs_top_q = copy(self.coeffs)

    def fit_info(self) -> None:
        """Plot fit and print coefficients."""
        self.print_report()
        self.plot_fit()
        self.plot_fit_iq()
        self.plot_model()
        plt.show()

    def plot_fit_iq(self, g_min=0, g_max=3, npoints=1000) -> None:
        """Plot the fits to the calibration dataset in the IQ plane."""
        g_fit = np.linspace(g_min, g_max, npoints)
        s_fit = self.model(g_fit)
        plt.figure()
        plt.plot(self.s.real, self.s.imag, ".", label="Measured")
        plt.plot(s_fit.real, s_fit.imag, label="Fit")
        plt.xlabel("In-phase RF signal")
        plt.ylabel("Out-of-phase RF signal")
        plt.legend()

    def plot_fit(self, g_min=0, g_max=3, npoints=1000) -> None:
        """Plot the RF signal vs DC conductance and the fit from the model."""
        g_fit = np.linspace(g_min, g_max, npoints)
        s_fit = self.model(g_fit)
        plt.figure()
        plt.plot(self.g, self.s.real, ".", label="In-phase measured")
        plt.plot(self.g, self.s.imag, ".", label="Out-of-phase measured")
        plt.plot(g_fit, s_fit.real, "-", label="In-phase fit")
        plt.plot(g_fit, s_fit.imag, "-", label="Out-of-phase fit")
        plt.xlabel("Measured DC conductance (e²/h)")
        plt.ylabel("RF signal")
        plt.legend()

    def plot_model(self, *, show=True) -> matplotlib.figure.Figure:
        """Plot the modeled conductance vs. the measured DC conductance."""
        dataset = self.calibrated_dataset()
        g_model = dataset[self.g_name]
        w, h = plt.figaspect(1)
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
        ax0 = plt.subplot2grid((2, 1), (0, 0), colspan=1, rowspan=1)
        ax0.scatter(self.g, g_model.real, c="k")
        ax0.set_ylabel("Estimated Re(g)")

        ax0.grid(0.25)

        ax1.axhline(0, c="k", lw=1)
        ax1.scatter(self.g, g_model.imag, c="r")
        ax1.set_xlabel("Measured DC conductance (e²/h)")
        ax1.set_ylabel("Estimated Im(g)")
        ax1.grid(0.25)

        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def inverse_model(self, s: NDArray[complex]) -> NDArray[float]:
        """Fit a Moebius transformation to the conductance and RF signal."""
        s0, a, b = pack_complex_coeffs(self.coeffs)
        return inverse_model(s, s0, a, b)

    def model(self, g: NDArray[float]) -> NDArray[complex]:
        """Calculate modeled RF signal as a function of conductance.

        Parameters
        ----------
        g : 1D array-like
            DC conductance.
        """
        s0, a, b = pack_complex_coeffs(self.coeffs)
        return model(g, s0, a, b)

    def calibrated_dataset(
        self,
        scale_factor: Optional[complex] = None,
        dataset: Optional[xr.Dataset] = None,
    ) -> xr.Dataset:
        """Calculate conductance from an RF dataset and the calibration.

        Parameters
        ----------
        scale_factor : complex, optional
            Scale factor to multiply the RF signal by before applying the calibration.
        dataset : Optional[xr.Dataset], optional
            Dataset with complex signal
            like  ``dataset[self.x_name] + 1.0j * dataset[self.y_name]`` if None
            `self.s` is used.

        Returns
        -------
        calibrated_dataset : xarray.Dataset
            A copy of the original dataset with an added conductance variable.
        """
        scale_factor = scale_factor or 1
        if dataset is None:
            s = self.s
            dataset = self.dataset
        else:
            s = dataset[self.x_name] + 1.0j * dataset[self.y_name]
        s_values = s * scale_factor
        g_values = self.inverse_model(s_values)
        dims = dataset[self.x_name].dims
        calibrated_dataset = dataset.assign({self.g_name: (dims, g_values.data)})
        calibrated_dataset[self.g_name].attrs["long_name"] = "Conductance"
        calibrated_dataset[self.g_name].attrs["units"] = "e²/h"
        return calibrated_dataset

    def optimize_scale_factor(
        self,
        dataset: Optional[xr.Dataset] = None,
        p0: NDArray[float] = None,
        plot: bool = True,
    ) -> complex:
        """Rotate an RF signal dataset to match the calibration dataset.

        Parameters
        ----------
        dataset : Optional[xr.Dataset], optional
            Dataset with complex signal
            like  ``dataset[self.x_name] + 1.0j * dataset[self.y_name]`` if None
            `self.s` is used.
        p0 : 1D array-like
            Initial values for optimization.
        plot : bool
            If True, plot the original calibration dataset and the scaled RF signal.
        """
        if p0 is None:
            p0 = [1, 0]
        if dataset is None:
            s = self.s
        else:
            s = np.ravel(dataset[self.x_name] + 1.0j * dataset[self.y_name])
        result = minimize(self.cost, args=(s,), x0=p0)
        r, t = result.x
        scale_factor = r * np.exp(1.0j * t)
        rotated_s = scale_factor * s
        if plot:
            plt.plot(rotated_s.real, rotated_s.imag, ".")
            plt.plot(s.real, s.imag, ".")
        return scale_factor

    def cost(self, params: Tuple[float, float], s: NDArray[complex]) -> float:
        """Cost for scale factor optimization.

        Parameters
        ----------
        params : tuple
            Tuple of parameters (x, y).
        s : 1D complex np.ndarray
            Complex RF signal.

        Returns
        -------
        cost : float
        """
        scale_factor = complex(*params)
        rf_signal_scaled = scale_factor * s
        g_model = self.inverse_model(rf_signal_scaled)
        cost = np.mean(g_model.imag ** 2)
        return cost

    def optimize_phase(
        self, dataset: Optional[xr.Dataset] = None, plot=True
    ) -> complex:
        """Rotate an RF signal dataset to match the calibration dataset.

        Parameters
        ----------
        dataset : Optional[xr.Dataset], optional
            Dataset with complex signal
            like  ``dataset[self.x_name] + 1.0j * dataset[self.y_name]`` if None
            `self.s` is used.
        p0 : 1D array-like
            Initial values for optimization.
        plot : bool
            If True, plot the original calibration dataset and the scaled RF signal.
        """
        p0 = [0.0]
        if dataset is None:
            s = self.s
        else:
            s = np.ravel(dataset[self.x_name] + 1.0j * dataset[self.y_name])
        result = minimize(lambda t: self.cost((np.cos(t), np.sin(t)), s), x0=p0)
        t = result.x
        scale_factor = np.exp(1.0j * t)
        rotated_s = scale_factor * s
        if plot:
            plt.plot(rotated_s.real, rotated_s.imag, ".")
            plt.plot(self.s.real, self.s.imag, ".")
        return scale_factor


def bootstrap(
    function: Callable[[NDArray[(2, ...), complex]], NDArray[complex]],
    samples: NDArray[(2, ...), complex],
    weights: Optional[NDArray[float]] = None,
    delta_threshold: float = 1e-3,
    max_iterations: int = 10000,
    confidence: float = 0.95,
    **kwargs: Any,
) -> Tuple[NDArray[complex], NDArray[float], NDArray[float]]:
    """Boostrap a function over a set of samples adaptively.

    The resampling will stop when the length of the confidence interval
    converges, i.e. its relative change with an additional resampling is below
    ``delta_threshold``.

    Parameters
    ----------
    function : callable
        A function of the samples. Must accept a 2D array-like
        where each sample is a row, and each column is a variable, and
        must return a 1D array-like.
    samples : 2D array-like
        Each row corresponds to one sample, and each column is a variable.
    weights : 1D array-like
        Probability weights for each sample for resampling with replacement.
    delta_threshold : float
        Relative change of confidence interval length for establishing
        convergence of the bootstrap.
    max_iterations : int
        Maximum number of resampling iterations.
    confidence : float
        Confidence threshold for confidence intervals.
    """
    n = samples.shape[0]
    means = function(samples)
    outputs = []
    last_q_range = np.zeros(len(means))
    indices = np.arange(n)
    bottom_quantile = (1.0 - confidence) / 2
    top_quantile = 1 - bottom_quantile
    min_iterations = int(1 / (1 - confidence))
    for _ in range(max_iterations):
        new_indices = choice(indices, size=n, p=weights)
        new_samples = samples[new_indices, :]
        output = function(new_samples, **kwargs)  # type: ignore
        outputs.append(output)
        new_q_range = np.quantile(outputs, q=top_quantile, axis=0) - np.quantile(
            outputs, q=bottom_quantile, axis=0
        )
        if len(outputs) >= min_iterations:
            rel_delta = np.abs(new_q_range - last_q_range) / new_q_range
            if np.all(rel_delta <= delta_threshold):
                break
        last_q_range = new_q_range
    return (
        means,
        np.quantile(outputs, q=bottom_quantile, axis=0),
        np.quantile(outputs, q=top_quantile, axis=0),
    )


def pack_complex_coeffs(coeffs: Iterable[float]) -> List[complex]:
    """Pack a list of real and imaginary parts into a list of complex coeffs."""
    return [complex(*x) for x in toolz.itertoolz.partition(2, coeffs)]


def unpack_complex_coeffs(*complex_coeffs: Iterable[complex]) -> List[float]:
    """Unpack complex_coeffs into a list of real and imaginary parts."""
    real_coeffs = []
    for x in complex_coeffs:
        real_coeffs.append(x.real)
        real_coeffs.append(x.imag)
    return real_coeffs


def fit_nonlinear(
    g: NDArray[float], s: NDArray[complex]
) -> Tuple[complex, complex, complex]:
    """Perform a nonlinear fit to the measured data.

    Parameters
    ----------
    g : 1D complex np.ndarray
        The DC conductance.
    s : 1D complex np.ndarray
        Complex RF signal.

    Returns
    -------
    tuple
        A tuple of complex fitting coefficient (s0, a, b).
    """
    p_least_sq = fit_least_sq(g, s)
    p_0 = unpack_complex_coeffs(*p_least_sq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # TODO: the line below prints a lot of warnings:
        # ```
        # .../scipy/optimize/minpack.py:475: RuntimeWarning: Number of calls
        # to function has reached maxfev = 1400.
        # ```
        # Increasing `maxfev=20_000` increases the runtime significantly.
        # In principle `least_squares(lambda p: model_residuals(g, s, p), p_0, method="lm").x`
        # should give the same result, however, in initial testing this doesn't
        # appear to be the case.
        params, _ = leastsq(lambda p: model_residuals(g, s, p), p_0)
    params = pack_complex_coeffs(params)
    return params


def model_residuals(
    g: NDArray[float], s: NDArray[complex], p: NDArray[float]
) -> NDArray[float]:
    """Calculate the absolute residuals of the conductance to RF model."""
    p_complex = pack_complex_coeffs(p)
    s_model = model(g, *p_complex)
    residuals = np.abs(s_model - s)
    return residuals


def fit_least_sq(
    g: NDArray[float], s: NDArray[complex]
) -> Tuple[complex, complex, complex]:
    """Perform a least squares fit to the measured data.

    Parameters
    ----------
    g : 1D complex np.ndarray
        The DC conductance.
    s : 1D complex np.ndarray
        Complex RF signal.

    Returns
    -------
    tuple
        A tuple of complex fitting coefficient (s0, a, b).
    """
    # coefficient matrix for least-squares fitting
    m = np.vstack([g, -np.ones(len(g)), -s * g]).T
    coeffs, _, _, _ = np.linalg.lstsq(m, s, rcond=None)
    s0 = coeffs[1]
    a = coeffs[0] / coeffs[1]
    b = coeffs[2]
    return s0, a, b


def model(g: NDArray[float], s0: complex, a: complex, b: complex) -> NDArray[complex]:
    """Calculate modeled RF signal as a function of conductance.

    Parameters
    ----------
    g : 1D array-like
        DC conductance.
    s0 : complex
        Complex fitting coefficient.
    a : complex
        Complex fitting coefficient.
    b : complex
        Complex fitting coefficient.
    """
    return s0 * (a * g - 1.0) / (b * g + 1.0)


def inverse_model(
    s: NDArray[complex], s0: complex, a: complex, b: complex
) -> NDArray[complex]:
    """Calculate modeled conductance as a function of RF signal.

    Parameters
    ----------
    s : 1D complex np.ndarray
        Complex RF signal.
    s0 : complex
        Complex fitting coefficient.
    a : complex
        Complex fitting coefficient.
    b : complex
        Complex fitting coefficient.

    Returns
    -------
    g : 1D complex np.ndarray
        The conductance.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # If the denominator becomes zero, ignore the warning
        g = (s + s0) / (-b * s + s0 * a)
    return g


def goodness_of_fit(
    g: NDArray[float], s: NDArray[complex], s0: complex, a: complex, b: complex
) -> float:
    """Calculate R² for the modeled and measured conductances.

    Parameters
    ----------
    g : 1D complex np.ndarray
        The DC conductance.
    s : 1D complex np.ndarray
        Complex RF signal.
    s0 : complex
        Complex fitting coefficient.
    a : complex
        Complex fitting coefficient.
    b : complex
        Complex fitting coefficient.

    Returns
    -------
    r : 1D real np.ndarray
        The fit "goodness" R².
    """
    g_modeled = inverse_model(s, s0, a, b)
    r_sq = np.corrcoef(g, g_modeled)[0, 1] ** 2
    r_sq_real = np.real(r_sq)
    return r_sq_real
