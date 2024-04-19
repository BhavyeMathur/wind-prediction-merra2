import datetime
from typing import Callable

import numpy as np

import cmasher as cmr
import cartopy.crs as projections
from cartopy.feature.nightshade import Nightshade

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter

_PLATE_CARREE = projections.PlateCarree()

mpl.rcParams['figure.dpi'] = 175


class MeteorologicalPlot:
    def __init__(self, data, title: str = "Meteorological Plot", output: str = None,
                 diverging: bool = False, maxmin_quantile: float = 0, title_size=9, label_size=7, **kwargs):
        self._backend = "matplotlib"
        self._data = data

        self._title = title
        self._output = output
        self._fig = None

        self.title_size = title_size
        self.label_size = label_size
        self.diverging = diverging
        self.maxmin_quantile = maxmin_quantile

        self._plot_kwargs = kwargs

    def _plot_mpl(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def _create_mpl_plot(self) -> plt.Figure:
        raise NotImplementedError()

    def _setup_mpl_plot(self) -> None:
        self._fig.suptitle(self._title, fontsize=self.title_size, y=0.95)

    def plot(self, *args, **kwargs) -> None:
        if "vmin" not in self._plot_kwargs and "vmax" not in self._plot_kwargs:
            vmin, vmax = self._get_vmin_and_vmax()
            self._plot_kwargs["vmin"] = vmin
            self._plot_kwargs["vmax"] = vmax

        if self._backend == "matplotlib":
            self._fig = self._create_mpl_plot()
            self._setup_mpl_plot()
            self._plot_mpl(*args, **self._plot_kwargs, **kwargs)

            if self._output:
                plt.savefig(f"generated/contours/{self._output}.png", dpi=300)

        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    def show(self):
        if self._backend == "matplotlib":
            plt.show()
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    def _get_vmin_and_vmax(self):
        vmin = np.quantile(self._data, self.maxmin_quantile)
        vmax = np.quantile(self._data, 1 - self.maxmin_quantile)

        if not self.diverging:
            return vmin, vmax

        if abs(vmin) > vmax:  # possible when vmin < 0
            return vmin, -vmin
        return -vmax, vmax


class GraticulePlot(MeteorologicalPlot):
    def __init__(self, data, *args, projection=_PLATE_CARREE, coastlines: bool = False, nightshade: bool = False,
                 colorbar: bool = True, **kwargs):
        super().__init__(data, *args, **kwargs)

        lats = np.linspace(-90, 90, self._data.shape[0])
        lons = np.linspace(-180, 180, self._data.shape[1])
        self._mesh = np.meshgrid(lons, lats)

        self._ax = None

        self.projection = projection

        self._colorbar = colorbar
        self._coastlines = coastlines
        self._nightshade = nightshade
        self._nightshade_dt = datetime.datetime(year=1980, month=1, day=1)

        self._line_width = 0.1
        self._line_alpha = 0.3

    def _create_mpl_plot(self) -> plt.Figure:
        fig = plt.figure(figsize=self._get_figsize(), layout=self._get_layout())
        # plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0)

        if self._projection is None:
            self._ax = plt.gca()
        else:
            self._ax = fig.add_subplot(projection=self._projection)

        self._ax.spines[:].set_color("#fff")

        if self._coastlines:
            self._ax.coastlines(linewidth=1, alpha=0.2)

        if self._nightshade:
            self._ax.add_feature(self._nightshade)

        self._draw_graticules()

        return fig

    def _draw_graticules(self):
        xticks = tuple(range(-150, 151, 50))
        yticks = tuple(range(-80, 81, 20))

        if self._projection is None:
            self._ax.grid(True, which="both", linestyle="dashed", linewidth=self._line_width)

            self._ax.tick_params(labelsize=self.label_size)
            self._ax.xaxis.set_major_formatter(FormatStrFormatter("%d°"))
            self._ax.yaxis.set_major_formatter(FormatStrFormatter("%d°"))
        else:
            gl = self._ax.gridlines(crs=_PLATE_CARREE, linewidth=self._line_width, linestyle='-', color="white",
                                    alpha=self._line_alpha, draw_labels=True)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlocator = mticker.FixedLocator(xticks)
            gl.ylocator = mticker.FixedLocator(yticks)

            gl.xlabel_style["size"] = self.label_size
            gl.ylabel_style["size"] = self.label_size

    def _get_figsize(self) -> tuple[float, float]:
        if isinstance(self._projection, (projections.Robinson, projections.Mollweide)):
            return 7, 4

        if isinstance(self._projection, (projections.Mercator,)):
            return 8, 7

        if isinstance(self._projection, (projections.AzimuthalEquidistant,)):
            return 6, 6

        if isinstance(self._projection, projections.PlateCarree):
            return 8, 4

        return 8, 5

    def _get_layout(self) -> str:
        if isinstance(self._projection, (projections.Robinson, projections.Mollweide)):
            return "none"
        return "tight"

    def _draw_color_bar(self, cmap, vmin, vmax, **_) -> None:
        if not self._colorbar:
            return

        norm = mpl.colors.Normalize(vmin, vmax)
        plot = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plot.set_array([])

        cbar = self._fig.colorbar(plot, ax=self._ax, fraction=0.03, pad=0.03)
        cbar.outline.set_linewidth(0.05)

        cbar.ax.tick_params(labelsize=7, right=False, direction="in")

    @property
    def projection(self):
        return self._projection

    @projection.setter
    def projection(self, value: Callable | projections.Projection):
        if value is None:
            self._projection = None
            self._plot_kwargs.pop("transform", None)
            self._plot_kwargs["aspect"] = "auto"

        elif isinstance(value, projections.Projection):
            self._projection = value
            self._plot_kwargs["transform"] = _PLATE_CARREE
            self._plot_kwargs.pop("aspect", None)

        else:
            try:
                self.projection = value()
                return
            except TypeError:
                pass

            raise ValueError("Projection must be a cartopy projection")


class ContourfPlot(GraticulePlot):
    def _plot_mpl(self, levels=25, **kwargs) -> None:
        self._ax.contourf(*self._mesh, self._data, levels=levels, **kwargs)
        self._draw_color_bar(**kwargs)


class ContourPlot(GraticulePlot):
    def _plot_mpl(self, levels=20, **kwargs) -> None:
        self._ax.contour(*self._mesh, self._data, levels=levels, **kwargs)
        self._draw_color_bar(**kwargs)


class ImagePlot(GraticulePlot):
    def _plot_mpl(self, **kwargs) -> None:
        self._ax.imshow(self._data, origin="lower", extent=(-180, 180, -90, 90), **kwargs)
        self._draw_color_bar(**kwargs)
