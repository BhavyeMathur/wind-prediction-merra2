import sys
from typing import Callable

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as projections

from modules.maths.barometric import height_from_pressure
from modules.era5.variables import AtmosphericVariable

from .text import format_pressure, format_time, format_latitude, format_longitude

if sys.platform == "darwin":
    mpl.use("TkAgg")
mpl.rcParams["figure.dpi"] = 150

_PLATE_CARREE = projections.PlateCarree()


class ImagePlot2D:
    def __init__(self, variable: AtmosphericVariable, indices: list, projection=None):
        self._variable = variable
        self._dset = variable[*indices]
        self._data = self._dset.to_dataarray().values
        self._indices = indices.copy()

        assert len(self._dset.dims) == 2, "Only 2D slices of data are supported"

        self._axes = tuple(self._dset.dims)[::-1]
        self._slice = tuple(key for key in self._dset.coords.keys() if key not in self._axes)

        self._kwargs = {}

        self._fig: None | plt.Figure = None
        self._ax: None | plt.Axes = None
        self.projection = projection

        self._has_secondary_axis = False

        self.show = plt.show
        self._reorder_axes()
        self._data = self._reshape_data(self._data)

    def _get_title(self) -> str:
        title = f"{self._variable.title} ({self._variable.axes_unit})"
        if self._axes == ("longitude", "latitude"):
            lev = int(self._dset['level'].values)
            return f"{title} at {lev} hPa {format_pressure(lev)} on {format_time(self._dset['time'].values)}"

        elif self._axes == ("longitude", "level"):
            return (f"{title} at {format_latitude(float(self._dset['latitude'].values))}"
                    f" on {format_time(self._dset['time'].values)}")

        elif self._axes == ("latitude", "level"):
            return (f"{title} at {format_longitude(float(self._dset['longitude'].values))}"
                    f" on {format_time(self._dset['time'].values)}")
        return title

    def _reorder_axes(self) -> None:
        if self._axes == ("level", "longitude"):
            self._axes = ("longitude", "level")
            self._reorder_axes()

        if self._axes == ("level", "latitude"):
            self._axes = ("latitude", "level")
            self._reorder_axes()

        assert (self._projection is not None and self._axes == ("longitude", "latitude")) or self._projection is None, \
            "Invalid projection for data type"

    def _reshape_data(self, data):
        data = data[0, ::-1]

        if self._axes[0] == "longitude":
            return np.roll(data, data.shape[1] // 2, axis=1)
        if self._axes[0] == "latitude":
            return data[:, ::-1]

        return data

    def _get_figsize(self) -> tuple[float, float]:
        if self._axes == ("longitude", "latitude"):
            return 8, 5
        elif self._axes == ("longitude", "level"):
            self._has_secondary_axis = True
            return 9, 3
        elif self._axes == ("latitude", "level"):
            self._has_secondary_axis = True
            return 9, 3

    def _get_axes_lims(self) -> tuple[tuple[float, float], tuple[float, float]]:
        if self._axes == ("longitude", "latitude"):
            return (-180, 180), (-90, 90)

        elif self._axes == ("longitude", "level"):
            return (-180, 180), (1000, 150)

        elif self._axes == ("latitude", "level"):
            return (-90, 90), (1000, 150)

        return ((round(float(self._dset[self._axes[0]].min())), round(float(self._dset[self._axes[0]].max()))),
                (round(float(self._dset[self._axes[1]].min())), round(float(self._dset[self._axes[1]].max()))))

    def plot(self, **kwargs) -> None:
        self._fig = plt.figure(figsize=self._get_figsize())
        self._fig.suptitle(self._get_title(), fontsize=9, y=0.9 if self._has_secondary_axis else 0.95)

        self._create_axes()
        self._fig.tight_layout()

        obj = self._plot_data(**(self._kwargs | {"cmap": self._variable.cmap} | kwargs))
        self._draw_colorbar(obj)

    def _draw_colorbar(self, obj) -> None:
        if self._has_secondary_axis:
            cbar = self._fig.colorbar(obj, fraction=0.06, pad=0.02, anchor=(0, 0), aspect=12, location="top",
                                      format=f"%.1f")
            cbar.ax.tick_params(labelsize=5)
        else:
            cbar = self._fig.colorbar(obj, fraction=0.06, pad=0.02, format=f"%.0f{self._variable.unit}")
            cbar.ax.tick_params(labelsize=6)

        cbar.outline.set_linewidth(0)
        cbar.ax.tick_params(width=0, direction="in")

    def _create_axes(self) -> None:
        if self._projection is None:
            self._ax = plt.gca()
        else:
            self._ax = self._fig.add_subplot(projection=self._projection)

        if self._axes[1] == "level":
            self._ax.yaxis.set_inverted(True)
            self._ax.set_yscale("log")

            levels = [1000, 850, 700, 550, 400, 250, 150]
            self._ax.set_yticks(levels, minor=False)

            ax2 = self._ax.twinx()
            ax2.set_yscale("log")
            ax2.set_ylim(150, 1000)
            ax2.yaxis.set_inverted(True)

            levels = 10 ** np.linspace(np.log10(150), np.log10(1000), 7, endpoint=True)
            ax2.set_yticks(levels, minor=False)
            ax2.set_yticks([], minor=True)
            ax2.set_yticklabels(map(lambda lev: f"{round(height_from_pressure(lev * 100) / 1000, 1)} km", levels))

            ax2.set_frame_on(False)
            ax2.yaxis.set_tick_params(width=0, labelsize=5)

        self._ax.set_frame_on(False)
        self._draw_grid()

        self._ax.set_xticks([], minor=True)
        self._ax.set_yticks([], minor=True)

        self._ax.xaxis.set_tick_params(width=0, labelsize=7)
        self._ax.yaxis.set_tick_params(width=0, labelsize=6 if self._axes[1] == "level" else 7)

        xunit = AtmosphericVariable[self._axes[0]].axes_unit
        yunit = AtmosphericVariable[self._axes[1]].axes_unit
        self._ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f"%d{xunit}"))
        self._ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%d{yunit}"))

    def _draw_grid(self) -> None:
        if self._projection is not None:
            gl = self._ax.gridlines(crs=_PLATE_CARREE, linewidth=0.15, linestyle=':', color="white", alpha=0.3,
                                    draw_labels=True, x_inline=False, y_inline=False)
            gl.top_labels = False
            gl.right_labels = False

            gl.xlabel_style["size"] = 7
            gl.ylabel_style["size"] = 7

            self._ax.set_extent((-180, 180, -90, 90))

            # noinspection PyProtectedMember
            if isinstance(self._projection, projections._RectangularProjection):
                gl.xlocator = mticker.FixedLocator(tuple(range(-180, 181, 60)))
            else:
                gl.xlocator = mticker.FixedLocator([])
            gl.ylocator = mticker.FixedLocator(tuple(range(-80, 81, 20)))
            return

        if self._axes == ("longitude", "latitude"):
            axis = "both"
        elif self._axes[0] in {"latitude", "longitude"}:
            axis = "x"
        elif self._axes[1] in {"latitude", "longitude"}:
            axis = "y"
        else:
            axis = None

        self._ax.grid(True, which="both", axis=axis, linestyle=":", linewidth=0.15)

    def _plot_data(self, **kwargs):
        xlims, ylims = self._get_axes_lims()
        kwargs = dict(extent=(*xlims, *ylims), origin="lower", interpolation="nearest") | kwargs
        return self._ax.imshow(self._data, **kwargs)

    def _get_x(self):
        xlims, _ = self._get_axes_lims()
        return np.linspace(*xlims, self._data.shape[1])

    def _get_y(self):
        _, ylims, = self._get_axes_lims()
        return np.linspace(*ylims, self._data.shape[0])

    def _get_mesh(self):
        x = self._get_x()
        y = self._get_y()
        return np.meshgrid(x, y)

    def _get_uv_resolution(self, type_: str) -> tuple[int, int]:
        if type_ == "barb":
            if self._axes == ("longitude", "latitude"):
                return 60, 30
            return 100, 20

        elif type_ == "stream":
            if self._axes == ("longitude", "latitude"):
                return 120, 60
            raise NotImplementedError("Streamplot only implemented for longitude & latitude plots")

        elif type_ == "quiver":
            if self._axes == ("longitude", "latitude"):
                return 120, 60
            return 40, 25

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

    @property
    def projection(self) -> None | projections.Projection:
        return self._projection

    @projection.setter
    def projection(self, value: Callable | projections.Projection | None) -> None:
        if value is None:
            self._projection = value
            self._kwargs.pop("transform", None)
            self._kwargs["aspect"] = "auto"

        elif isinstance(value, projections.Projection):
            self._projection = value
            self._kwargs["transform"] = _PLATE_CARREE
            self._kwargs.pop("aspect", None)

        else:
            try:
                self.projection = value()
            except TypeError:
                raise ValueError("Projection must be a cartopy projection")


class Contour2D(ImagePlot2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kwargs["levels"] = 20

    def _plot_data(self, **kwargs):
        return self._ax.contour(*self._get_mesh(), self._data, **kwargs)


class Contourf2D(Contour2D):
    def _plot_data(self, **kwargs):
        return self._ax.contourf(*self._get_mesh(), self._data, **kwargs)


__all__ = ["ImagePlot2D", "Contour2D", "Contourf2D"]
