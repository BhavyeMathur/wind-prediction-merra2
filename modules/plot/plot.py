import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.contour
import matplotlib.ticker as mticker

import cmasher as cmr
import cartopy.crs as projections
from cartopy.feature.nightshade import Nightshade

import numpy as np

from modules.merra2.dataset import MERRA2Dataset
from .text import *

import sys
from datetime import datetime

if sys.platform == "darwin":
    import matplotlib

    matplotlib.use('TkAgg')

_PLATE_CARREE = projections.PlateCarree()


def _get_vmin_and_vmax(data, diverging: bool = False, color_quantile: float = 0, **_):
    if not diverging:
        return None, None

    vmin = np.quantile(data, color_quantile)
    vmax = np.quantile(data, 1 - color_quantile)

    if abs(vmin) > vmax:  # only possible when vmin < 0
        return vmin, -vmin
    return -vmax, vmax


def _get_figsize_from_projection(projection) -> tuple[float, float]:
    if isinstance(projection, (projections.Robinson, projections.Mollweide)):
        return 7, 4

    if isinstance(projection, (projections.Mercator,)):
        return 8, 7

    if isinstance(projection, (projections.AzimuthalEquidistant,)):
        return 6, 6

    if isinstance(projection, projections.PlateCarree):
        return 8, 4

    return 8, 5


def _get_layout_from_projection(projection) -> str:
    if isinstance(projection, (projections.Robinson, projections.Mollweide)):
        return "none"

    return "tight"


def _get_latlon_mesh(shape):
    lats = np.linspace(-90, 90, shape[0])
    lons = np.linspace(-180, 180, shape[1])
    return np.meshgrid(lons, lats)


def _draw_color_bar(fig: plt.Figure, plot: matplotlib.cm.ScalarMappable) -> None:
    if isinstance(plot, matplotlib.contour.ContourSet):
        # drawing a continuous color bar (without discrete contour levels)
        norm = matplotlib.colors.Normalize(vmin=plot.cvalues.min(), vmax=plot.cvalues.max())
        plot = plt.cm.ScalarMappable(norm=norm, cmap=plot.cmap)
        plot.set_array([])

    cbar = fig.colorbar(plot, ax=fig.gca(), fraction=0.03, pad=0.03)
    cbar.outline.set_linewidth(0.05)

    cbar.ax.tick_params(labelsize=7, right=False, direction="in")


def _setup_figure(ax: plt.Axes, title: str, title_size: float = 9):
    ax.set_title(title, fontsize=title_size)
    ax.spines[:].set_color("#fff")


def _add_earth_figure_grid(ax: plt.Axes, projection, labelsize=7, linewidth=0.1):
    xticks = tuple(range(-150, 151, 50))
    yticks = tuple(range(-80, 81, 20))

    if projection is None:
        ax.grid(True, which="both", linestyle="dashed", linewidth=linewidth)

        ax.tick_params(labelsize=labelsize)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d°"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d°"))
    else:
        gl = ax.gridlines(crs=_PLATE_CARREE, linewidth=linewidth, linestyle='-', color="white", alpha=0.3,
                          draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(xticks)
        gl.ylocator = mticker.FixedLocator(yticks)

        gl.xlabel_style["size"] = labelsize
        gl.ylabel_style["size"] = labelsize


def _new_earth_figure(title: str, output: list[str], projection=None,
                      coastlines: bool = False, nightshade: bool = False, **_):
    fig = plt.figure(figsize=_get_figsize_from_projection(projection),
                     layout=_get_layout_from_projection(projection))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0)

    if projection is None:
        ax = fig.gca()
    else:
        ax = fig.add_subplot(projection=projection)

        if coastlines:
            ax.coastlines(linewidth=1, alpha=0.2)

        if nightshade:
            ax.add_feature(Nightshade(datetime(year=1980, month=1, day=1, hour=12), alpha=0.2))

        output.append(projection.__class__.__name__.lower())

    _setup_figure(ax, title)
    _add_earth_figure_grid(ax, projection)

    return fig, ax


def _set_plot_kwargs(data: np.ndarray, plot_type: str, **kwargs):
    plot_kwargs = {}

    if (projection := kwargs.get("projection")) is None:
        plot_kwargs["aspect"] = "auto"
    elif isinstance(projection, projections.Projection):
        plot_kwargs["transform"] = _PLATE_CARREE
    else:
        raise ValueError("Non-Cartopy projections not yet supported")

    vmin, vmax = _get_vmin_and_vmax(data, **kwargs)
    plot_kwargs["vmin"] = vmin
    plot_kwargs["vmax"] = vmax

    plot_kwargs["cmap"] = kwargs.get("cmap", cmr.ocean)

    if plot_type.startswith("contour"):
        plot_kwargs["levels"] = kwargs.get("levels", 20)

    return plot_kwargs


def plot_dataset(dataset: MERRA2Dataset, time: None | str = None, lev: None | int = None, latitude: None | int = None,
                 data_transform=lambda data: data, **kwargs) -> None:
    data = dataset.load(time=time, lev=lev, lat=latitude)
    data = data_transform(data)

    # Latitude vs Longitude contour
    if len(data.shape) == 2:
        output = _plot_latitude_longitude_contour(data, format_title(dataset.variable, time, lev),
                                                  format_output(dataset.variable, time, lev), **kwargs)
    else:
        raise ValueError("Data shape incompatible with plot")

    plt.savefig(f"generated/contours/{'-'.join(output)}", dpi=300)
    plt.show()


def _plot_latitude_longitude_contour(data: np.ndarray, title, output: list[str],
                                     plot_type: str | list[str] = "image", cbar: bool = True, **kwargs) -> list[str]:
    fig, ax = _new_earth_figure(title, output, **kwargs)
    plot_kwargs = _set_plot_kwargs(data, plot_type, **kwargs)

    if plot_type == "contourf":
        plotted_data = ax.contourf(*_get_latlon_mesh(data.shape), data, **plot_kwargs)
        output.append("contourf")

    elif plot_type == "contour":
        plotted_data = ax.contour(*_get_latlon_mesh(data.shape), data, **plot_kwargs)
        output.append("contour")

    elif "image" in plot_type:
        plotted_data = ax.imshow(data, origin="lower", extent=(-180, 180, -90, 90), **plot_kwargs)

        if "contour" in plot_type:
            output.append("image-contour")

            plot_kwargs |= {"colors": ["black"], "cmap": None, "linewidths": [0.5], "alpha": 0.2}
            ax.contour(*_get_latlon_mesh(data.shape), data, **plot_kwargs)

    else:
        raise ValueError(f"Unknown plot type {plot_type!r}")

    if cbar:
        _draw_color_bar(fig, plotted_data)

    return output
