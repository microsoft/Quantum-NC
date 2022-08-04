##
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MICROSOFT QUANTUM NON-COMMERCIAL License.
##
import abc
import inspect
import itertools
from contextlib import suppress
from typing import Dict, Optional, Tuple, Union

import xarray as xr
from typeguard import check_type

from topogap_protocol.datalake import PathLike


def check_data_array(
    data: Dict[str, xr.Dataset],
    name: str,
    std_dims: Tuple[str, ...],
    std_vars: Tuple[str, ...],
):
    # Check dataset is at least there
    if name not in data:
        print(f"Data does not contain dataset {name}.")
        return False
    dataset = data[name]

    # Check dataset is not empty
    if dataset is None:
        print(f"Dataset is empty: {name}")
        return False

    check = True

    # Check coordinate names
    coords = set(dataset.coords.keys())
    has_all_coords = all(d in coords for d in std_dims)
    if not has_all_coords:
        print(
            f"Dataset {name} doesn't have the required"
            f" ({std_dims}) coordinate names but has {coords}."
        )
        check = False
    extra_coords = coords - set(std_dims)
    if extra_coords:
        print(f"Dataset {name} has extra coordinate names: {extra_coords}.")
        check = True

    # Check data variables names
    data_vars = set(dataset.data_vars.keys())
    has_all_data_vars = all(d in data_vars for d in std_vars)
    if not has_all_data_vars:
        print(
            f"Dataset {name} doesn't have the required"
            f" ({std_vars}) variable names but has {data_vars}."
        )
        check = False
    extra_data_vars = data_vars - set(std_vars)
    if extra_data_vars:
        print(f"Dataset {name} has extra coordinate names: {extra_data_vars}.")
        check = True
    return check


class _RequireAttrsABCMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)

        obj._required_attrs = {}
        for type_ in cls.mro():
            with suppress(AttributeError):
                obj._required_attrs.update(type_.__annotations__)

        for name, type_ in obj._required_attrs.items():
            try:
                x = getattr(obj, name)
            except AttributeError:
                raise AttributeError(
                    f"Required attribute {name} not set in __init__."
                ) from None
            else:
                try:
                    check_type("", x, type_)
                except TypeError:
                    msg = f"The attribute '{name}' should be of type {type_}, not {type(x)}."
                    raise TypeError(msg)

        # Restore the signature of __init__
        sig = inspect.signature(cls.__init__)
        parameters = tuple(sig.parameters.values())
        __signature__ = sig.replace(parameters=parameters[1:])
        cls.__signature__ = __signature__
        return obj


class _AbstractData(abc.ABC, metaclass=_RequireAttrsABCMeta):
    folder: Optional[PathLike]
    data: Dict[str, xr.Dataset]

    @property
    def device_id(self) -> str:
        return str(self.folder.name).split("-")[0]

    def _info(self, which: str) -> Tuple[str, str, str, str, str, str]:
        return (
            f"{which} data of topological gap protocol.",
            "",
            f"Device ID     : {self.device_id}",
            f"Path to data  : {self.folder}",
            "",
        )

    def check_data(self) -> None:
        """Checks that all arrays exist and have the right names."""
        data_is_ready = [
            check_data_array(
                data=self.data,
                name=name,
                std_dims=self.get_standard_dim_names(),
                std_vars=self.get_standard_var_names(),
            )
            for name in self.get_standard_xarray_names()
        ]
        if all(data_is_ready):
            print(f"{self.__class__.__name__} is ready.")
        else:
            print("Data is not ready. See problems mentioned above.")

    @abc.abstractmethod
    def load_data(self) -> None:
        """Reduce data and extract RF signals."""

    @abc.abstractmethod
    def get_standard_dim_names(self) -> Tuple[str, ...]:
        """Standard names for the dimensions."""

    @abc.abstractmethod
    def get_standard_var_names(self) -> Tuple[str, ...]:
        """Standard names for the variables."""

    @abc.abstractmethod
    def get_standard_xarray_names(self) -> Tuple[str, ...]:
        """RF signal measured with left or right resonator."""


class AbstractCalibrationData(_AbstractData):
    def __str__(self) -> str:
        s = (
            *self._info("Calibration"),
            "\n",
            "Left side calibration data:",
            str(self.data["left"]),
            "\n",
            "Right side calibration data:",
            str(self.data["right"]),
        )
        return "\n".join(s)

    @staticmethod
    def get_standard_dim_names() -> Tuple[str, str, str]:
        return ("rc", "lc", "f")

    @staticmethod
    def get_standard_var_names() -> Tuple[str, str, str]:
        """Real part, imaginary part, conductance."""
        return ("re", "im", "g")

    @staticmethod
    def get_standard_xarray_names() -> Tuple[str, str]:
        """RF signal measured with left or right resonator."""
        return ("left", "right")


class AbstractRFData(_AbstractData):
    cross_sweeps: bool

    def __str__(self) -> str:
        s = self._info("RFData")
        for i, j in itertools.product(("left", "right"), repeat=2):
            s += (  # type: ignore
                f"{i.capitalize()} cutter, {j} bias scan:",
                f"Saved in: {self.nc_filename(i[0], j[0])}",
                str(self.data[f"{i[0]}c_{j[0]}b"]),
                "\n",
            )

        return "\n".join(s)

    @staticmethod
    def get_standard_dim_names() -> Tuple[str, str, str, str, str, str]:
        return ("p", "lb", "lc", "rb", "rc", "f")

    @staticmethod
    def get_standard_var_names() -> Tuple[str, str, str, str]:
        return ("rf_l_re", "rf_l_im", "rf_r_re", "rf_r_im")

    def get_standard_xarray_names(
        self,
    ) -> Union[Tuple[str, str, str, str], Tuple[str, str]]:
        if self.cross_sweeps:
            return ("lc_lb", "lc_rb", "rc_lb", "rc_rb")
        else:
            return ("lc_lb", "rc_rb")


class AbstractPhaseOneData(AbstractRFData):
    """Abstract base class for PhaseOneData."""

    @staticmethod
    def get_standard_var_names() -> Tuple[str, str, str, str]:
        return ("g_ll", "g_lr", "g_rl", "g_rr")

    def __str__(self) -> str:
        return super().__str__(self).replace("RFData", "PhaseOneData")


class AbstractPhaseTwoRegionData(_AbstractData):
    region_number: int

    def __str__(self) -> str:
        s = (
            *self._info("Phase two")[:-1],
            f"Region number : {self.region_number}",
            "",
            "\nLeft bias scan:",
            str(self.data["left"]),
            "\nRight bias scan:",
            str(self.data["right"]),
        )
        return "\n".join(s)

    @staticmethod
    def get_standard_dim_names() -> Tuple[str, str, str, str, str, str]:
        return ("p", "lb", "lc", "rb", "rc", "f")

    @staticmethod
    def get_standard_var_names() -> Tuple[str, str, str, str, str, str]:
        return ("g_ll", "g_lr", "g_rl", "g_rr", "i_l", "i_r")

    @staticmethod
    def get_standard_xarray_names() -> Tuple[str, str]:
        """Data measured scanning either left or right bias."""
        return ("left", "right")
