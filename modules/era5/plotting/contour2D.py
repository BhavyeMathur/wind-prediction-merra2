import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import cartopy.crs as projections
from cartopy.mpl.geoaxes import GeoAxes

from modules.maths.barometric import height_from_pressure
from modules.era5.variables import AtmosphericVariable
from modules.maths.util import minmax_norm
from modules.datetime import timedelta, DateTime

from .text import format_altitude, format_time, format_latitude, format_longitude
from .plotter import *

_PLATE_CARREE = projections.PlateCarree()


class _Contour2D:
    _has_secondary_axis = False
    _figsize = 9, 3
    _axes_lims: tuple[tuple[float, float], tuple[float, float]]
    _xunit = "°"
    _yunit = "°"
    _grid = "both"

    def __init__(self, variable: AtmosphericVariable | tuple[AtmosphericVariable, ...], indices: list):
        if isinstance(variable, tuple) and len(variable) == 1:
            variable = variable[0]

        self._variable = variable
        self._indices = indices.copy()

        if isinstance(variable, tuple):
            self._dset = xr.Dataset({var.name: var[*indices][var.name] for var in variable})
        else:
            self._dset = variable[*indices]
        assert len(self._dset.dims) == 2, "Only 2D slices of data are supported"

        self._data = self._dset.to_dataarray().values
        self._data = self._reshape_data(self._data)
        self._plotter = ImagePlot2D(self._axes_lims)

        self._fig: None | plt.Figure = None
        self._ax: None | plt.Axes = None

        self.show = plt.show
        self.save = plt.savefig

    def _get_title(self) -> str:
        if isinstance(self._variable, tuple):
            title = ", ".join(var.title for var in self._variable)
        else:
            title = f"{self._variable.title} ({self._variable.unit})"

        return title + self._get_title_slice_substring()

    def _get_title_position(self) -> float:
        return 0.9 if self._has_secondary_axis else 0.95

    def _get_title_slice_substring(self) -> str:
        return ""

    def _reshape_data(self, data: np.ndarray) -> np.ndarray:
        data = data.squeeze()

        if data.ndim == 3:  # If image data, normalise all channels to [0, 1]
            data = np.dstack([minmax_norm(var) for var in data])
            if data.shape[-1] == 2:  # add 3rd channel (RGB) to data
                data = np.dstack([np.zeros(data.shape[:-1]), data])

        return data

    def _create_axes(self) -> None:
        self._ax = plt.gca()
        self._ax.set_frame_on(False)

        self._ax.grid(True, which="both", axis=self._grid, linestyle=":", linewidth=0.15)

        self._ax.set_xticks([], minor=True)
        self._ax.set_yticks([], minor=True)

        self._ax.xaxis.set_tick_params(width=0, labelsize=7)
        self._ax.yaxis.set_tick_params(width=0, labelsize=7)

        self._ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f"%d{self._xunit}"))
        self._ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%d{self._yunit}"))

    def plot(self, **kwargs):
        self._fig = plt.figure(figsize=self._figsize)
        self._fig.suptitle(self._get_title(), fontsize=9, y=self._get_title_position())

        self._create_axes()
        self._fig.tight_layout()

        kwargs = {} if isinstance(self._variable, tuple) else {"cmap": self._variable.cmap} | kwargs
        obj = self._plotter.plot(self._ax, self._data, **kwargs)

        if isinstance(self._variable, AtmosphericVariable):
            self._draw_colorbar(obj)
        self._add_map()

    def _get_map_projection(self) -> projections.Projection:
        return projections.Robinson()

    def _get_map_extent(self) -> str | tuple[float, float, float, float]:
        return "global"

    def _plot_map_slice(self, ax) -> None:
        raise NotImplementedError()

    def _add_map(self) -> None:
        ax: GeoAxes = plt.axes((0.81, 0.82, 0.13, 0.13), projection=self._get_map_projection())

        # noinspection PyTypeChecker
        ax.gridlines(xlocs=[-150, -100, -50, 0, 50, 100, 150], linewidth=0.15)
        ax.coastlines(linewidth=0.1)

        ax.spines[:].set_linewidth(0.1)
        ax.spines[:].set_color("#000")

        if (extent := self._get_map_extent()) == "global":
            ax.set_global()
        else:
            ax.set_extent(extent, crs=_PLATE_CARREE)

        self._plot_map_slice(ax)

    def _draw_colorbar(self, obj) -> None:
        fmt = f"%.{max(0, 2 - round(np.log10(np.max(self._data) - np.min(self._data))))}f"

        if self._has_secondary_axis:
            cbar = self._fig.colorbar(obj, fraction=0.06, pad=0.02, anchor=(0, 0), aspect=12, location="top",
                                      format=fmt)
            labelsize = 5
        else:
            cbar = self._fig.colorbar(obj, fraction=0.06, pad=0.02, format=f"{fmt}{self._variable.unit}")
            labelsize = 6

        cbar.outline.set_linewidth(0)
        cbar.ax.tick_params(width=0, direction="in", labelsize=labelsize)

    def _get_x(self):
        return np.linspace(*self._axes_lims[0], self._data.shape[1])

    def _get_y(self):
        return np.linspace(*self._axes_lims[1], self._data.shape[0])

    def _get_uv_resolution(self, type_: str) -> tuple[int, int]:
        raise NotImplementedError()

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


class _LatLon2D(_Contour2D):
    _figsize = 8, 5
    _axes_lims = (-180, 180), (-90, 90)

    def _get_title_slice_substring(self) -> str:
        lev = int(self._dset['level'].values)
        return (f" at {lev} hPa {format_altitude(lev)}"
                f" on {format_time(self._dset['time'].values)}")

    def _reshape_data(self, data):
        data = super()._reshape_data(data)[::-1]
        return np.roll(data, data.shape[1] // 2, axis=1)

    def _add_map(self) -> None:
        return

    def _get_uv_resolution(self, type_: str) -> tuple[int, int]:
        if type_ == "barb":
            return 60, 30
        elif type_ == "stream":
            return 120, 60
        elif type_ == "quiver":
            return 120, 60


class _Lev2D(_Contour2D):
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

    def _get_uv_resolution(self, type_: str) -> tuple[int, int]:
        raise NotImplementedError("UV Stream/barb/quiver plots only implemented for longitude & latitude plots")


class _LatLev2D(_Lev2D):
    _axes_lims = (-90, 90), (1000, 150)

    def __init__(self, variable: AtmosphericVariable | tuple[AtmosphericVariable, ...], indices: list):
        super().__init__(variable, indices)
        self._lon = float(self._dset["longitude"].values)

    def _get_title_slice_substring(self) -> str:
        return (f" at {format_longitude(float(self._dset['longitude'].values))}"
                f" on {format_time(self._dset['time'].values)}")

    def _get_map_projection(self) -> projections.Projection:
        return projections.Robinson(central_longitude=self._lon)

    def _plot_map_slice(self, ax) -> None:
        ax.plot([self._lon, self._lon], [-90, 90], transform=_PLATE_CARREE, linewidth=0.4, color="red")

    def _reshape_data(self, data):
        return super()._reshape_data(data)[::-1, ::-1]


class _LonLev2D(_Lev2D):
    _axes_lims = (-180, 180), (1000, 150)

    def __init__(self, variable: AtmosphericVariable | tuple[AtmosphericVariable, ...], indices: list):
        super().__init__(variable, indices)
        self._lat = float(self._dset["latitude"].values)

    def _get_title_slice_substring(self) -> str:
        return (f" at {format_latitude(float(self._dset['latitude'].values))}"
                f" on {format_time(self._dset['time'].values)}")

    def _reshape_data(self, data):
        data = super()._reshape_data(data)[::-1]
        return np.roll(data, data.shape[1] // 2, axis=1)

    def _get_map_extent(self) -> str | tuple[float, float, float, float]:
        if self._lat == 0:
            return "global"
        elif self._lat > 0:
            return -180, 180, 0, 90
        else:
            return -180, 180, -90, 0

    def _plot_map_slice(self, ax) -> None:
        ax.plot([-180, 180], [self._lat, self._lat], transform=_PLATE_CARREE, linewidth=0.4, color="red")


class _Time2D(_Contour2D):
    _has_secondary_axis = True
    _grid = "y"

    def _create_axes(self) -> None:
        super()._create_axes()
        self._ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


class _TimeLon(_Time2D):
    _axes_lims = (DateTime(1, 1, 0), DateTime(12, 31, 23)), (-180, 180)

    def __init__(self, variable: AtmosphericVariable | tuple[AtmosphericVariable, ...], indices: list):
        super().__init__(variable, indices)
        self._lat = float(self._dset["latitude"].values)

    def _get_title_slice_substring(self) -> str:
        lev = int(self._dset['level'].values)
        return (f" at {lev} hPa {format_altitude(lev)}"
                f" along {format_latitude(self._lat)}")

    def _reshape_data(self, data):
        data = super()._reshape_data(data).T
        return np.roll(data, data.shape[0] // 2, axis=0)

    def _plot_map_slice(self, ax) -> None:
        ax.plot([-180, 180], [self._lat, self._lat], transform=_PLATE_CARREE, linewidth=0.4, color="red")


class _TimeLat(_Time2D):
    _axes_lims = (DateTime(1, 1, 0), DateTime(12, 31, 23)), (-90, 90)

    def __init__(self, variable: AtmosphericVariable | tuple[AtmosphericVariable, ...], indices: list):
        super().__init__(variable, indices)
        self._lon = float(self._dset["longitude"].values)

    def _get_title_slice_substring(self) -> str:
        lev = int(self._dset['level'].values)
        return (f" at {lev} hPa {format_altitude(lev)}"
                f" along {format_longitude(self._lon)}")

    def _plot_map_slice(self, ax) -> None:
        ax.plot([self._lon, self._lon], [-90, 90], transform=_PLATE_CARREE, linewidth=0.4, color="red")

    def _get_map_projection(self) -> projections.Projection:
        return projections.Robinson(central_longitude=self._lon)

    def _reshape_data(self, data):
        return super()._reshape_data(data).T[::-1]


def plot_contour2D(variable: AtmosphericVariable | tuple[AtmosphericVariable, ...], indices: list) -> _Contour2D:
    if isinstance(variable, AtmosphericVariable):
        time, lev, lat, lon = variable.get_full_index(indices)
    else:
        time, lev, lat, lon = variable[0].get_full_index(indices)

    if time is not None:
        if lev is not None:  # lat and lon are None
            return _LatLon2D(variable, indices)
        if lat is not None:  # lev and lon are None
            return _LonLev2D(variable, indices)
        if lon is not None:   # lev and lat are None
            return _LatLev2D(variable, indices)

    indices[0] = slice("TAVG-01-01 00:00", "TAVG-12-31 12:00", timedelta(days=1))
    if lev is not None:  # time is None
        if lat is not None:
            return _TimeLon(variable, indices)
        if lon is not None:
            return _TimeLat(variable, indices)

    raise NotImplementedError()


__all__ = ["plot_contour2D"]
