from __future__ import annotations

import numpy as np

import os

from modules.merra2.datafile import MERRA2File
from modules.merra2.util import get_merra_time_index_from_time
from modules.datetime import parse_datetime_depr, date_as_number


class MERRA2Dataset:

    __slots__ = ["_path", "_variable", "_is_float16", "_files", "_file_from_date", "_dimensions",
                 "_has_multiple_years", "_has_multiple_days"]

    _instances: dict[(str, str), MERRA2Dataset] = {}

    def __init__(self, path: str, variable: str, is_float16: bool = False) -> None:
        self._path: str = os.path.abspath(path)
        self._variable: str = variable

        self._is_float16 = is_float16
        self._has_multiple_years = False
        self._has_multiple_days = False

        self._files: list[MERRA2File] = []
        self._file_from_date: dict[MERRA2File.DATE_TYPE, MERRA2File] = {}
        self._find_files()

        self._dimensions = dict(zip(self._files[0].dims, self._files[0].shape))

    def __repr__(self) -> str:
        return f"MERRA2Dataset({self._variable!r}) at {self._path!r} ({len(self._files)} file/s found)"

    def __hash__(self) -> int:
        return hash((self._path, self._variable))

    def _find_files(self) -> None:
        if os.path.isfile(self._path):
            self._add_data_file(self._path)
            self._path = os.path.dirname(self._path)
        else:
            self._find_files_in_directory()

    def _find_files_in_directory(self) -> None:
        for filename in os.listdir(self._path):
            if filename.endswith(".nc4"):
                self._add_data_file(f"{self._path}/{filename}")

        self._validate_files()

        self._files = sorted(self._files, key=lambda file: date_as_number(*file.date))

        self._has_multiple_years = len(set(map(MERRA2File.year.fget, self._files))) != 1
        self._has_multiple_days = len(set(map(lambda file: date_as_number(*file.date), self._files))) != 1

    def _add_data_file(self, filepath: str) -> None:
        file = MERRA2File(filepath, self._is_float16)
        if self._variable in file:
            self._files.append(file)
            self._file_from_date[file.date] = file

    def _validate_files(self) -> None:
        assert len(self._files) > 0, "Found no files for data set"
        assert len(set(map(MERRA2File.ndims.fget, self._files))) == 1, "number of dimensions in dataset files differ"
        assert len(set(map(MERRA2File.dims.fget, self._files))) == 1, "dimensions in dataset files differ"
        assert len(set(map(MERRA2File.shape.fget, self._files))) == 1, "size of dimensions in dataset files differ"

    @property
    def path(self) -> str:
        """
        Returns:
            The folder in which the files for this data set are located
        """
        return self._path

    @property
    def variable(self) -> str:
        """
        Returns:
            The variable code this data set refers to
        """
        return self._variable

    @property
    def is_float16(self) -> bool:
        """
        Returns:
            Whether the data is stored as compressed float16s
        """
        return self._is_float16

    @property
    def files(self) -> tuple[str, ...]:
        """
        Returns:
            List of filepaths in this data set
        """
        return tuple(map(MERRA2File.filepath.fget, self._files))

    @property
    def ndims(self) -> int:
        """
        Returns:
            Number of dimensions in the data set
        """
        return sum(dim != 1 for dim in self._dimensions.values())

    @property
    def dims(self) -> tuple[str, ...]:
        """
        Returns:
            Names of the dimensions in the data set
        """
        return tuple(self._dimensions.keys())

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns:
            Sizes of the dimensions in the data set
        """
        return len(self._files), *self._dimensions.values()

    def _get_file_at_date(self, day: None | int, month: None | int, year: None | int) -> MERRA2File:
        if year is None and self._has_multiple_years:
            raise ValueError("Ambiguous reference to year. Specify which year to load the data for.")

        # (day is None) == (month is None) in all cases
        if day is None and self._has_multiple_days:  # and month is None
            raise ValueError("Ambiguous reference to date. Specify which date to load the data for.")

        year = self._files[0].year if year is None else year
        month = self._files[0].month if month is None else month
        day = self._files[0].day if day is None else day

        try:
            return self._file_from_date[(day, month, year)]  # time = '* DD-MM-YYYY'
        except KeyError:
            raise ValueError(f"No file found for '{day}-{month}-{year}'")

    def load(self, time: None | str | list[str] = None, lev: None | int = None, lat: None | int = None):
        if isinstance(time, list):
            return np.concatenate([self.load(t, lev, lat) for t in time])

        minute, hour, day, month, year = parse_datetime_depr(time)

        time = get_merra_time_index_from_time(minute, hour, self._dimensions["time"])
        file = self._get_file_at_date(day, month, year)

        return file.load(self._variable, time, lev, lat)
