import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as projections
from cartopy.mpl.geoaxes import GeoAxes

from era5.variables import AtmosphericVariable
from era5.maths.util import minmax_norm

from .plotter import *


class MetPlot:
    _has_secondary_axis = False
    _figsize = 9, 3
    _axes_lims: tuple[tuple[float, float], tuple[float, float]]
    _xunit = "°"
    _yunit = "°"
    _grid = "both"
    _plotter = ImagePlot2D
    _colorbar = True

    def __init__(self, variable: AtmosphericVariable | tuple[AtmosphericVariable, ...], indices: list,
                 transform=lambda data: data):
        if isinstance(variable, tuple) and len(variable) == 1:
            variable = variable[0]

        self._variable = variable
        self._indices = indices.copy()

        if isinstance(variable, tuple):
            self._dset = xr.Dataset({var.name: (dset := var[*indices])[var.name] for var in variable})
            self._dset.attrs = dset.attrs
        else:
            self._dset = variable[*indices]

        self._data = self._dset.to_dataarray().values
        self._data = self._reshape_data(self._data)
        self._plotter = self._plotter(self._axes_lims)

        self._transform = transform

        self._fig: None | plt.Figure = None
        self._ax: None | plt.Axes = None

        self.save = plt.savefig
        self._plotted = False

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
        if not isinstance(data, np.ndarray):
            data = np.array(data)
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

    def plot(self, data=None, **kwargs):
        if self._fig is None:
            self._fig = plt.figure(figsize=self._figsize)
            self._fig.suptitle(self._get_title(), fontsize=9, y=self._get_title_position())

            self._create_axes()
            self._fig.tight_layout()

        kwargs = {} if isinstance(self._variable, tuple) else {"cmap": self._variable.cmap} | kwargs

        if data is None:
            data = self._data
        else:
            data = self._reshape_data(data)
        obj = self._plotter.plot(self._ax, self._transform(data), **kwargs)

        if isinstance(self._variable, AtmosphericVariable) and self._colorbar:
            self._draw_colorbar(obj)
        self._add_map()

        self._plotted = True

    def show(self):
        if not self._plotted:
            self.plot()
        plt.show()

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
            ax.set_extent(extent, crs=projections.PlateCarree())

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


class PlotVersusLongitude(MetPlot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lat = self._dset["latitude"].values

    def _plot_map_slice(self, ax) -> None:
        ax.plot([-180, 180], [self._lat, self._lat], transform=projections.PlateCarree(), linewidth=0.4)

    def _get_map_extent(self) -> str | tuple[float, float, float, float]:
        if isinstance(self._lat, np.ndarray):
            return "global"

        if self._lat == 0:
            return "global"
        elif self._lat > 0:
            return -180, 180, 0, 90
        else:
            return -180, 180, -90, 0

    def _reshape_data(self, data):
        axis = list(self._dset.dims).index("longitude")
        data = super()._reshape_data(data)
        return np.roll(data, data.shape[axis] // 2, axis=axis)


class PlotVersusLatitude(MetPlot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lon = float(self._dset["longitude"].values)

    def _plot_map_slice(self, ax) -> None:
        ax.plot([self._lon, self._lon], [-90, 90], transform=projections.PlateCarree(), linewidth=0.4)

    def _get_map_projection(self) -> projections.Projection:
        return projections.Robinson(central_longitude=self._lon)
