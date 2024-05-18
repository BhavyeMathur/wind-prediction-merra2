import numpy as np

import matplotlib.dates as mdates
import cartopy.crs as projections

from era5.util.datetime import DateTime

from .metplot import MetPlot
from .text import format_altitude, format_latitude, format_longitude


class _Time2D(MetPlot):
    _has_secondary_axis = True
    _grid = "y"

    def _create_axes(self) -> None:
        super()._create_axes()
        self._ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


class _TimeLon2D(_Time2D):
    _axes_lims = (DateTime(1, 1, 0), DateTime(12, 31, 23)), (-180, 180)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lat = float(self._dset["latitude"].values)

    def _get_title_slice_substring(self) -> str:
        lev = int(self._dset['level'].values)
        return (f" at {lev} hPa {format_altitude(lev)}"
                f" along {format_latitude(self._lat)}")

    def _reshape_data(self, data):
        data = super()._reshape_data(data).T
        return np.roll(data, data.shape[0] // 2, axis=0)

    def _plot_map_slice(self, ax) -> None:
        ax.plot([-180, 180], [self._lat, self._lat], transform=projections.PlateCarree(), linewidth=0.4, color="red")


class _TimeLat2D(_Time2D):
    _axes_lims = (DateTime(1, 1, 0), DateTime(12, 31, 23)), (-90, 90)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lon = float(self._dset["longitude"].values)

    def _get_title_slice_substring(self) -> str:
        lev = int(self._dset['level'].values)
        return (f" at {lev} hPa {format_altitude(lev)}"
                f" along {format_longitude(self._lon)}")

    def _plot_map_slice(self, ax) -> None:
        ax.plot([self._lon, self._lon], [-90, 90], transform=projections.PlateCarree(), linewidth=0.4, color="red")

    def _get_map_projection(self) -> projections.Projection:
        return projections.Robinson(central_longitude=self._lon)

    def _reshape_data(self, data):
        return super()._reshape_data(data).T[::-1]
