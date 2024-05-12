import xarray as xr

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

    @datetime_func("datetime")
    def open(self, datetime) -> xr.DataArray:
        """
        Selects an xarray DataArray from a Dataset opened from a datetime
        """
        # Check dictionary cache. If DataArray already opened, return that.
        if datetime in self._datasets:
            return self._datasets[datetime]

        # Otherwise, open the DataArray and add it to cache before returning it.
        self._datasets[datetime] = AtmosphericVariable._open_ds(datetime)[self.name]
        return self._datasets[datetime]

    @staticmethod
    @datetime_func("datetime")
    def _open_ds(datetime) -> xr.Dataset:
        """
        Opens an xarray DataSet source file from a datetime
        """
        # Check dictionary cache. If DataSet already opened, return that.
        if datetime in AtmosphericVariable._datasets:
            return AtmosphericVariable._datasets[datetime]

        # Otherwise, open the DataSet and add it to cache before returning it.
        AtmosphericVariable._datasets[datetime] = xr.open_dataset(f"{ERA5}/ERA5-{datetime}.nc")
        return AtmosphericVariable._datasets[datetime].expand_dims({"time": [datetime]})


# time, level, latitude, longitude
class AtmosphericVariable4D(AtmosphericVariable):
    def __getitem__(self, item):
        level = None
        latitude = None
        longitude = None

        # Extract time, level, latitude, and longitude indices from argument
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

        # If the time index is a slice, extract data from each time in slice and concatenate result
        if isinstance(time, slice):
            data = []
            for dt in datetime_range(time.start, time.stop, time.step if time.step else datetime.timedelta(hours=1)):
                data.append(self[dt, level, latitude, longitude])
            return xr.concat(data, "time")

        # Open DataArray and select appropriate data before returning
        ds = self.open(time)
        if level is None:
            return ds
        elif latitude is None:
            return ds.sel(level=level)
        elif longitude is None:
            return ds.sel(level=level, latitude=latitude)
        else:
            return ds.sel(level=level, latitude=latitude, longitude=longitude)

    @staticmethod
    def get_vlims(level: int) -> tuple[float, float]:
        """
        Returns 'limits' of the data at a particular level to use in plotting
        """
        raise NotImplementedError()


# time, latitude, longitude
class AtmosphericVariable3D(AtmosphericVariable):
    @staticmethod
    def get_vlims() -> tuple[float, float]:
        """
        Returns 'limits' of the data to use in plotting
        """
        raise NotImplementedError()


# latitude, longitude
class AtmosphericVariable2D(AtmosphericVariable):
    @staticmethod
    def get_vlims() -> tuple[float, float]:
        """
        Returns 'limits' of the data to use in plotting
        """
    raise NotImplementedError()


class Temperature(AtmosphericVariable4D):
    def __init__(self):
        super().__init__(name="temperature", unit="K",
                         cmap=["#d7e4fc", "#5fb1d4", "#4e9bc8", "#466ae1", "#6b1966",
                               "#952c5e", "#d12d3e", "#fa7532", "#f5d25f"])

    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return -40, 40
        elif level == 150:
            return -70, -40

        return super().get_vlims(level)


class UWind(AtmosphericVariable4D):
    def __init__(self):
        super().__init__(name="u_component_of_wind", unit="m/s", cmap=cmr.ocean)

    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return -15, 15

        return super().get_vlims(level)


class VWind(AtmosphericVariable4D):
    def __init__(self):
        super().__init__(name="v_component_of_wind", unit="m/s", cmap=cmr.ocean)

    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return -15, 15

        return super().get_vlims(level)


class VerticalVelocity(AtmosphericVariable4D):
    def __init__(self):
        super().__init__(name="vertical_velocity", unit="Pa/s", cmap="RdBu")

    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return -1.5, 1.5

        return super().get_vlims(level)


class WindDirection(AtmosphericVariable4D):
    def __init__(self):
        super().__init__(name="wind_direction", unit="°", cmap="twilight")

    @staticmethod
    def get_vlims(_):
        return -180, 180


class WindSpeed(AtmosphericVariable4D):
    def __init__(self):
        super().__init__(name="wind_speed", unit="m/s", cmap=cmr.ocean)

    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return 0, 16

        return super().get_vlims(level)


__all__ = [Temperature, UWind, VWind, VerticalVelocity, WindDirection, WindSpeed]
