import numpy as np

import matplotlib.ticker as mticker

from era5.maths.barometric import height_from_pressure

from .metplot import MetPlot, PlotVersusLongitude, PlotVersusLatitude
from .text import format_time, format_latitude, format_longitude


class _Lev2D(MetPlot):
    _has_secondary_axis = True
    _xunit = "°"
    _yunit = "mb"
    _grid = "x"

    def _create_axes(self) -> None:
        super()._create_axes()

        self._ax.yaxis.set_inverted(True)
        self._ax.set_yscale("log")

        self._ax.set_yticks([1000, 850, 700, 550, 400, 250, 150], minor=False)
        self._ax.set_yticks([], minor=True)

        self._ax.yaxis.set_tick_params(labelsize=6)
        self._ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%d {self._yunit}"))

        self._create_secondary_axis()

    def _create_secondary_axis(self):
        ax2 = self._ax.twinx()

        ax2.yaxis.set_inverted(True)
        ax2.set_ylim(150, 1000)
        ax2.set_yscale("log")

        levels = 10 ** np.linspace(np.log10(150), np.log10(1000), 7, endpoint=True)
        ax2.set_yticks(levels, minor=False)
        ax2.set_yticks([], minor=True)
        ax2.set_yticklabels(map(lambda lev: f"{round(height_from_pressure(lev * 100) / 1000, 1)} km", levels))

        ax2.set_frame_on(False)
        ax2.yaxis.set_tick_params(width=0, labelsize=5)


class _LatLev2D(PlotVersusLatitude, _Lev2D):
    _axes_lims = (-90, 90), (1000, 150)

    def _get_title_slice_substring(self) -> str:
        return (f" at {format_longitude(float(self._dset['longitude'].values))}"
                f" on {format_time(self._dset['time'].values)}")

    def _reshape_data(self, data):
        return super()._reshape_data(data)[::-1, ::-1]


class _LonLev2D(PlotVersusLongitude, _Lev2D):
    _axes_lims = (-180, 180), (1000, 150)

    def _get_title_slice_substring(self) -> str:
        return (f" at {format_latitude(float(self._dset['latitude'].values))}"
                f" on {format_time(self._dset['time'].values)}")

    def _reshape_data(self, data):
        data = super()._reshape_data(data)[::-1]
        return np.roll(data, data.shape[1] // 2, axis=1)
