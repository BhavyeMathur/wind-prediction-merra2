import matplotlib.dates as mdates

from era5.util.datetime import DateTime

from .metplot import MetPlot, PlotVersusLongitude, PlotVersusLatitude
from era5.variables.text import format_altitude, format_latitude, format_longitude


class _Time2D(MetPlot):
    _has_secondary_axis = True
    _grid = "y"

    def _create_axes(self) -> None:
        super()._create_axes()
        self._ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


class _TimeLon2D(PlotVersusLongitude, _Time2D):
    _axes_lims = (DateTime(1, 1, 0), DateTime(12, 31, 23)), (-180, 180)

    def _get_title_slice_substring(self) -> str:
        lev = int(self._dset['level'].values)
        return (f" at {lev} hPa {format_altitude(lev)}"
                f" along {format_latitude(self._lat)}")

    def _reshape_data(self, data):
        return super()._reshape_data(data).T


class _TimeLat2D(PlotVersusLatitude, _Time2D):
    _axes_lims = (DateTime(1, 1, 0), DateTime(12, 31, 23)), (-90, 90)

    def _get_title_slice_substring(self) -> str:
        lev = int(self._dset['level'].values)
        return (f" at {lev} hPa {format_altitude(lev)}"
                f" along {format_longitude(self._lon)}")

    def _reshape_data(self, data):
        return super()._reshape_data(data).T[::-1]
