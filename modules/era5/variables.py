from __future__ import annotations
from typing import Type

import xarray as xr

import cmasher as cmr
from matplotlib.colors import LinearSegmentedColormap, Colormap

from modules.datetime import datetime_func, DateTime, datetime_range, timedelta

ERA5 = "/Volumes/Seagate Hub/ERA5/wind"


class AtmosphericVariable:
    _datasets: dict[DateTime, xr.Dataset] = {}
    _variables: dict[str, Type[AtmosphericVariable]] = {}

    _name: str | None
    _unit: str | None
    _cmap: Colormap | str
    _dtype: str

    def __init__(self):
        self._datasets: dict[DateTime, xr.DataSet] = {}

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

    @datetime_func("dt")
    def open(self, dt) -> xr.Dataset:
        """
        Selects a xarray DataArray from a Dataset opened from a datetime
        """
        # Check dictionary cache. If DataArray already opened, return that.
        if dt in self._datasets:
            return self._datasets[dt]

        # Otherwise, open the DataArray and add it to cache before returning it.
        ds = AtmosphericVariable._open_ds(dt)
        darray = ds[[self.name]]
        darray.attrs |= ds.attrs
        self._datasets[dt] = darray

        return darray

    @staticmethod
    @datetime_func("datetime")
    def _open_ds(dt) -> xr.Dataset:
        """
        Opens a xarray DataSet source file from a datetime
        """
        # Check dictionary cache. If DataSet already opened, return that.
        if dt in AtmosphericVariable._datasets:
            return AtmosphericVariable._datasets[dt]

        # Otherwise, open the DataSet and add it to cache before returning it.
        AtmosphericVariable._datasets[dt] = xr.open_dataset(f"{ERA5}/ERA5-{dt}.nc")
        return AtmosphericVariable._datasets[dt].expand_dims({"time": [dt]})

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

        # Open DataArray and select appropriate data before returning
        ds = self.open(time)
        if latitude is None:
            ds = ds.sel(level=level)
        elif longitude is None:
            ds = ds.sel(level=level, latitude=latitude)
        else:
            ds = ds.sel(level=level, latitude=latitude, longitude=longitude)

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


class Time(AtmosphericVariable, name="time", unit="seconds", dtype="datetime"):
    pass


class Level(AtmosphericVariable, name="level", unit="hPa", dtype="int16"):
    pass


class Latitude(AtmosphericVariable, name="latitude", unit="degrees north"):
    pass


class Longitude(AtmosphericVariable, name="longitude", unit="degrees east"):
    pass


class Temperature(AtmosphericVariable4D, name="temperature", unit="K",
                  cmap=["#d7e4fc", "#5fb1d4", "#4e9bc8", "#466ae1", "#6b1966",
                        "#952c5e", "#d12d3e", "#fa7532", "#f5d25f"]):
    def __init__(self, celsius: bool = False):
        super().__init__()
        if celsius:
            self._unit = "°C"
        else:
            self._unit = "K"
        self._celsius = celsius

    def _getitem_post(self, ds: xr.Dataset | xr.DataArray) -> xr.DataArray:
        if self._celsius:
            return ds - 273.15  # Kelvin to Celsius
        return ds

    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return -40, 40
        elif level == 150:
            return -70, -40
        raise ValueError("Unknown level for value limits")


class UWind(AtmosphericVariable4D, name="u_component_of_wind", unit="m/s"):
    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return -15, 15
        raise ValueError("Unknown level for value limits")


class VWind(AtmosphericVariable4D, name="v_component_of_wind", unit="m/s"):
    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return -15, 15
        raise ValueError("Unknown level for value limits")


class VerticalVelocity(AtmosphericVariable4D, name="vertical_velocity", unit="Pa/s", cmap="RdBu"):
    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return -1.5, 1.5
        raise ValueError("Unknown level for value limits")


class WindDirection(AtmosphericVariable4D, name="wind_direction", unit="°", cmap="twilight"):
    @staticmethod
    def get_vlims(_):
        return -180, 180


class WindSpeed(AtmosphericVariable4D, name="wind_speed", unit="m/s"):
    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return 0, 16
        raise ValueError("Unknown level for value limits")


__all__ = ["Temperature", "UWind", "VWind", "VerticalVelocity", "WindDirection", "WindSpeed", "Level", "Latitude",
           "Longitude", "AtmosphericVariable"]


from .dataset import uncompress_dataset
