from __future__ import annotations
from typing import Type

import cmasher as cmr
from matplotlib.colors import LinearSegmentedColormap, Colormap

from modules.datetime import datetime_range, timedelta

import xarray as xr


class AtmosphericVariable:
    _variables: dict[str, Type[AtmosphericVariable]] = {}

    _name: str | None
    _unit: str | None
    _cmap: Colormap | str
    _dtype: str

    # noinspection PyMethodOverriding
    def __init_subclass__(cls, **kwargs):
        cls._name = kwargs.get("name", None)
        cls._unit = kwargs.get("unit", None)
        cls._cmap = kwargs.get("cmap", cmr.ocean)
        cls._dtype = kwargs.get("dtype", "float32")

        if isinstance(cls._cmap, list):
            cls._cmap = LinearSegmentedColormap.from_list(cls._name, cls._cmap)

        AtmosphericVariable._variables[cls._name] = cls

    @property
    def name(self):
        return self._name

    @property
    def unit(self):
        return self._unit

    @property
    def cmap(self):
        return self._cmap

    @property
    def dtype(self):
        return self._dtype

    @staticmethod
    def get_units(variable: str):
        return AtmosphericVariable._variables[variable].unit

    @staticmethod
    def get_dtype(variable: str):
        return AtmosphericVariable._variables[variable].dtype


# time, level, latitude, longitude
class AtmosphericVariable4D(AtmosphericVariable):
    def __getitem__(self, item) -> xr.DataArray:
        time, level, latitude, longitude = self._get_index(item)

        # If the time index is a slice, extract data from each time in slice and concatenate result
        if isinstance(time, slice):
            data = []
            for dt in datetime_range(time.start, time.stop, time.step if time.step else timedelta(hours=1)):
                data.append(self[dt, level, latitude, longitude])
            return xr.concat(data, "time")

        ds = open_variable(self.name, time)
        ds = select_slice(ds, level, latitude, longitude)
        ds = uncompress_dataset(ds)

        return self._getitem_post(ds)

    @staticmethod
    def _get_index(item):
        level = None
        latitude = None
        longitude = None

        if isinstance(item, tuple):
            time = item[0]
            if len(item) > 1:
                level = item[1]
            if len(item) > 2:
                latitude = item[2]
            if len(item) > 3:
                longitude = item[3]
        else:
            time = item

        return time, level, latitude, longitude

    def _getitem_post(self, ds: xr.Dataset | xr.DataArray) -> xr.DataArray:
        return ds


# time, latitude, longitude
class AtmosphericVariable3D(AtmosphericVariable):
    pass


# latitude, longitude
class AtmosphericVariable2D(AtmosphericVariable):
    pass


from modules.era5.dataset import uncompress_dataset, select_slice
from modules.era5.io import open_variable


__all__ = ["AtmosphericVariable", "AtmosphericVariable4D", "AtmosphericVariable3D", "AtmosphericVariable2D"]
