from __future__ import annotations

import xarray as xr
import numpy as np

import os


class MERRA2File:

    DATE_TYPE = tuple[int, int, int]

    __slots__ = ["_filepath", "_file", "_is_float16", "_year", "_month", "_day"]

    def __init__(self, filepath: str, is_float16: bool = False):
        assert filepath.endswith(".nc4"), "Only NetCDF4 files are supported"

        self._filepath = os.path.abspath(filepath)
        self._file: xr.Dataset = xr.open_dataset(self._filepath)

        date = self.filepath.removesuffix(".nc4").split(".")[-1]
        self._year = date[:4]
        self._month = int(date[4:6])
        self._day = int(date[6:])

        if self._year == "YAVG":
            self._year = 0

        self._is_float16 = is_float16

    def __del__(self) -> None:
        try:
            self._file.close()
        except AttributeError:
            pass

    def __repr__(self) -> str:
        return self._filepath + "\n" + self._file.__repr__()

    def __hash__(self) -> int:
        return hash(self._filepath)

    def __getitem__(self, item) -> xr.DataArray:
        return self._file[item]

    def __contains__(self, item):
        return item in self._file.variables

    def _uncompress(self, data: np.ndarray) -> np.ndarray:
        if self._is_float16:
            return data.view("float16").astype("float16")
        return data

    @property
    def ndims(self) -> int:
        """
        Returns:
            Number of dimensions in the file
        """
        return sum(dim != 1 for dim in self._file.sizes.values())

    @property
    def dims(self) -> tuple[str, ...]:
        """
        Returns:
            Names of the dimensions in the file
        """
        # noinspection PyTypeChecker
        return tuple(self._file.sizes.keys())

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns:
            Sizes of the dimensions in the file
        """
        return tuple(self._file.sizes.values())

    @property
    def filepath(self) -> str:
        """
        Returns:
            The filepath of this data file
        """
        return self._filepath

    @property
    def date(self) -> DATE_TYPE:
        """
        Returns:
            The date associated with this data file (day, month, year)
        """
        return self._day, self._month, self._year

    @property
    def day(self) -> int:
        """
        Returns:
            The day associated with the date of this data file
        """
        return self._day

    @property
    def month(self) -> int:
        """
        Returns:
            The month associated with the date of this data file
        """
        return self._month

    @property
    def year(self) -> int:
        """
        Returns:
            The year associated with the date of this data file
        """
        return self._year

    # TODO - Handle loading longitudes
    def load(self, variable: str,
             time: None | int = None, lev: None | int = None, lat: None | int = None) -> np.ndarray:

        indices = {"time": time, "lev": lev, "lat": lat, "lon": ":"}
        index = ", ".join((str(indices[dim]) for dim in self.dims)).replace("None", ":")

        data = np.array(eval(f"self._file['{variable}'][{index}]")).squeeze()
        data = self._uncompress(data)

        return data
