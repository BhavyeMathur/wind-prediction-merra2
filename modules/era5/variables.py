import xarray as xr
import numpy as np

import cmasher as cmr
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

import datetime
from modules.datetime import datetime_func, DateTime, datetime_range


ERA5 = "/Volumes/Seagate Hub/ERA5/wind"


class AtmosphericVariable:
    _datasets: dict[DateTime, xr.Dataset] = {}

    def __init__(self, name: str, unit: str, cmap: str | ListedColormap | list = cmr.ocean):
        self._name = name
        self._unit = unit

        if isinstance(cmap, list):
            self._cmap = LinearSegmentedColormap.from_list(self._name, cmap)
        else:
            self._cmap = cmap

        self._datasets: dict[DateTime, xr.DataArray] = {}

    @property
    def name(self):
        return self._name

    @property
    def unit(self):
        return self._unit

    @property
    def cmap(self):
        return self._cmap

    def get_vlims(self, level: int) -> tuple[float, float]:
        raise NotImplementedError()

    @datetime_func("datetime")
    def open(self, datetime) -> xr.DataArray:
        if datetime in self._datasets:
            return self._datasets[datetime]

        self._datasets[datetime] = AtmosphericVariable._open_ds(datetime)[self.name]
        return self._datasets[datetime]

    @staticmethod
    @datetime_func("datetime")
    def _open_ds(datetime) -> xr.Dataset:
        if datetime in AtmosphericVariable._datasets:
            return AtmosphericVariable._datasets[datetime]

        AtmosphericVariable._datasets[datetime] = xr.open_dataset(f"{ERA5}/ERA5-{datetime}.nc")
        return AtmosphericVariable._datasets[datetime]

    @staticmethod
    def _get_sliced_index_iterable(values, index: slice):
        values = list(values)

        start_idx = values.index(index.start)
        stop_idx = values.index(index.stop)
        step = -1 if index.start > index.stop else 1
        return values[start_idx:stop_idx:step][::index.step]


class AtmosphericVariable4D(AtmosphericVariable):
    def __getitem__(self, item):
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

        if isinstance(time, slice):
            data = []
            for dt in datetime_range(time.start, time.stop, time.step if time.step else datetime.timedelta(hours=1)):
                data.append(self[dt, level, latitude, longitude])
            return np.array(data)

        ds = self.open(time)

        if isinstance(level, slice):
            data = []

            for lev in AtmosphericVariable._get_sliced_index_iterable(ds.level.values, level):
                data.append(self[time, lev, latitude, longitude])
            return np.array(data)

        if isinstance(latitude, slice):
            data = []

            for lat in AtmosphericVariable._get_sliced_index_iterable(ds.latitude.values, latitude):
                data.append(self[time, level, lat, longitude])
            return np.array(data)

        if isinstance(longitude, slice):
            data = []

            for lon in AtmosphericVariable._get_sliced_index_iterable(ds.longitude.values, longitude):
                data.append(self[time, level, latitude, lon])
            return np.array(data)

        print(level, latitude)
        if level is None:
            return ds.values
        elif latitude is None:
            return ds.sel(level=level).values
        elif longitude is None:
            return ds.sel(level=level, latitude=latitude).values
        else:
            return ds.sel(level=level, latitude=latitude, longitude=longitude)


class AtmosphericVariable3D(AtmosphericVariable):
    pass


class AtmosphericVariable2D(AtmosphericVariable):
    pass


class Temperature(AtmosphericVariable4D):
    def __init__(self):
        super().__init__(name="temperature", unit="K",
                         cmap=["#d7e4fc", "#5fb1d4", "#4e9bc8", "#466ae1", "#6b1966",
                               "#952c5e", "#d12d3e", "#fa7532", "#f5d25f"])

    def get_vlims(self, level: int):
        if level == 1000:
            return -40, 40
        elif level == 150:
            return -70, -40

        return super().get_vlims(level)


class UWind(AtmosphericVariable4D):
    def __init__(self):
        super().__init__(name="u_component_of_wind", unit="m/s", cmap=cmr.ocean)

    def get_vlims(self, level: int):
        if level == 1000:
            return -15, 15

        return super().get_vlims(level)


class VWind(AtmosphericVariable4D):
    def __init__(self):
        super().__init__(name="v_component_of_wind", unit="m/s", cmap=cmr.ocean)

    def get_vlims(self, level: int):
        if level == 1000:
            return -15, 15

        return super().get_vlims(level)


class VerticalVelocity(AtmosphericVariable4D):
    def __init__(self):
        super().__init__(name="vertical_velocity", unit="Pa/s", cmap="RdBu")

    def get_vlims(self, level: int):
        if level == 1000:
            return -1.5, 1.5

        return super().get_vlims(level)


class WindDirection(AtmosphericVariable4D):
    def __init__(self):
        super().__init__(name="wind_direction", unit="°", cmap="twilight")

    def get_vlims(self, _):
        return -180, 180


class WindSpeed(AtmosphericVariable4D):
    def __init__(self):
        super().__init__(name="wind_speed", unit="m/s", cmap=cmr.ocean)

    def get_vlims(self, level: int):
        if level == 1000:
            return 0, 16

        return super().get_vlims(level)


__all__ = [Temperature, UWind, VWind, VerticalVelocity, WindDirection, WindSpeed]
