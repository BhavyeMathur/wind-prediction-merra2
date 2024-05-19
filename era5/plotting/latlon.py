import numpy as np

from era5.variables import AtmosphericVariable

from .text import format_altitude, format_time
from .metplot import MetPlot


class _MapContour(MetPlot):
    _figsize = 8, 5
    _axes_lims = (-180, 180), (-90, 90)

    def _get_title_slice_substring(self) -> str:
        lev = int(self._dset['level'].values)
        return (f" at {lev} hPa {format_altitude(lev)}"
                f" on {format_time(self._dset['time'].values, self._dset.attrs['is_tavg'])}")

    def _reshape_data(self, data):
        data = super()._reshape_data(data)[::-1]
        return np.roll(data, data.shape[1] // 2, axis=1)

    def add_map(self) -> None:
        return

    @staticmethod
    def _get_uv_resolution(type_: str) -> tuple[int, int]:
        if type_ == "barb":
            return 60, 30
        elif type_ == "stream":
            return 120, 60
        elif type_ == "quiver":
            return 120, 60

    def _get_uv_plot_data(self, type_: str, u, v, resolution: int | tuple[int, int] | None = None) -> tuple:
        if resolution is None:
            resolution = self._get_uv_resolution(type_)
        if isinstance(resolution, int):
            resolution = (resolution, resolution)

        xres, yres = resolution

        x = sorted(self._get_x())
        y = sorted(self._get_y())
        xstep = len(x) // xres
        ystep = len(y) // yres

        u = self._reshape_data(u.slice(self._indices))[::ystep, ::xstep]
        v = self._reshape_data(v.slice(self._indices))[::ystep, ::xstep]
        x = x[::xstep]
        y = y[::ystep]
        return x, y, u, v

    def add_streamlines(self, u: AtmosphericVariable, v: AtmosphericVariable, resolution=None, **kwargs) -> None:
        self._ax.streamplot(*self._get_uv_plot_data("stream", u, v, resolution),
                            **(dict(linewidth=0.2, color="#fff", density=3, arrowsize=0.5) | kwargs))

    def add_barbs(self, u: AtmosphericVariable, v: AtmosphericVariable, resolution=None, **kwargs) -> None:
        self._ax.barbs(*self._get_uv_plot_data("barb", u, v, resolution),
                       **(dict(linewidth=0.2, color="#fff", length=3.5) | kwargs))

    def add_quiver(self, u: AtmosphericVariable, v: AtmosphericVariable, resolution: int = None, **kwargs) -> None:
        self._ax.quiver(*self._get_uv_plot_data("quiver", u, v, resolution),
                        **(dict(linewidth=0.2, color="#fff") | kwargs))
