from __future__ import annotations
from typing import Type, Hashable, Final

import numpy as np
import xarray as xr

import threading
from tqdm import tqdm

import cmasher as cmr
from matplotlib.colors import LinearSegmentedColormap, Colormap

from era5.util.datetime import datetime_range, timedelta


class _AtmosphericVariableMetaclass(type):
    _variables: dict[Hashable, _AtmosphericVariableMetaclass]

    name: Final[str | None] = None
    title: Final[str | None] = None
    unit: Final[str | None] = None
    cmap: Final[Colormap | str] = cmr.ocean
    dtype: Final[str] = "float32"

    def __getitem__(self, item) -> _AtmosphericVariableMetaclass:
        return self._variables[item]


class AtmosphericVariable(metaclass=_AtmosphericVariableMetaclass):
    _variables: dict[Hashable, Type[AtmosphericVariable]] = {}
    _instances: dict[Hashable, AtmosphericVariable] = {}

    def __init__(self):
        AtmosphericVariable._instances[self.name] = self

    # noinspection PyMethodOverriding, PyFinal
    def __init_subclass__(cls, **kwargs):
        cls.name = kwargs.get("name", None)
        cls.title = kwargs.get("title", None if cls.name is None else cls.name.replace("_", " ").title())
        cls.unit = kwargs.get("unit", None)
        cls.cmap = kwargs.get("cmap", cmr.ocean)

        cls.dtype = kwargs.get("dtype", "float32")
        cls._requires = kwargs.get("requires", cls.name)

        if isinstance(cls.cmap, list):
            cls.cmap = LinearSegmentedColormap.from_list(cls.name, cls.cmap)

        if cls.name is None:
            return
        if cls.name in AtmosphericVariable._variables:
            raise RuntimeError(f"Cannot create multiple instances of variable '{cls.name}'")
        AtmosphericVariable._variables[cls.name] = cls

    def __getitem__(self, item) -> xr.Dataset:
        raise NotImplementedError()

    def slice(self, indices: list | tuple) -> np.ndarray:
        data = self[*indices]
        return data.to_dataarray().values

    def get_vlims(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def get(variable: Hashable) -> AtmosphericVariable:
        return AtmosphericVariable._instances[variable]

    @staticmethod
    def get_full_index(item) -> tuple[int, int, int, int]:
        raise NotImplementedError()


# time, level, latitude, longitude
class AtmosphericVariable4D(AtmosphericVariable):
    def __getitem__(self, item) -> xr.Dataset:
        time, level, latitude, longitude = self.get_full_index(item)

        # If the time index is a slice, extract data from each time in slice and concatenate result
        if isinstance(time, slice):
            threads = []
            data = {}
            dsets = []

            # TODO restructure some of this code
            dts = datetime_range(time.start, time.stop, time.step if time.step else timedelta(hours=1))
            for dt in dts:
                dsets.append(open_variable(self._requires, dt))

            def read_file_thread(i):
                dset = dsets[i]
                dset = select_slice(dset, level, latitude, longitude)
                dset = uncompress_dataset(dset)

                val = self._getitem_post(dset)

                if isinstance(val, xr.Dataset):
                    data[i] = val
                    return
                if not isinstance(val, dict):
                    val = {self.name: val}

                data[i] = xr.Dataset(val, coords=dset.coords, attrs=dset.attrs)

            for ds in range(len(dsets)):
                thread = threading.Thread(target=read_file_thread, args=(ds,))
                threads.append(thread)
                thread.start()

            for thread in tqdm(threads):
                thread.join()

            return xr.concat([data[i] for i in range(len(dsets))], "time")

        if time is None:
            return self["TAVG-01-01 00:00":"TAVG-12-31 23:00", level, latitude, longitude]

        ds = open_variable(self._requires, time)
        ds = select_slice(ds, level, latitude, longitude)
        ds = uncompress_dataset(ds)

        vals = self._getitem_post(ds)

        if isinstance(vals, xr.Dataset):
            return vals
        if not isinstance(vals, dict):
            vals = {self.name: (ds.dims, vals)}

        return xr.Dataset(vals, coords=ds.coords, attrs=ds.attrs)

    def get_vlims(self, level: int):
        raise NotImplementedError()

    def _getitem_post(self, ds: xr.Dataset) -> np.ndarray | xr.DataArray | xr.Dataset:
        return ds

    @staticmethod
    def get_full_index(item):
        level = None
        latitude = None
        longitude = None

        if isinstance(item, (tuple, list)):
            time = item[0]
            if len(item) > 1:
                level = item[1]
            if len(item) > 2:
                latitude = item[2]
            if len(item) > 3:
                longitude = item[3]
        else:
            time = item

        if isinstance(longitude, (int, float)) and longitude < 0:
            longitude %= 360

        return time, level, latitude, longitude


# time, latitude, longitude
class AtmosphericVariable3D(AtmosphericVariable):
    pass


# latitude, longitude
class AtmosphericVariable2D(AtmosphericVariable):
    pass


from era5.dataset import uncompress_dataset, select_slice
from era5.io import open_variable


__all__ = ["AtmosphericVariable", "AtmosphericVariable4D", "AtmosphericVariable3D", "AtmosphericVariable2D"]
