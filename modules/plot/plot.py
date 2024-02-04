import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import cmasher as cmr
import cartopy.crs as projections

import numpy as np

from modules.merra2.dataset import MERRA2Dataset
from .text import *

import sys

if sys.platform == "darwin":
    import matplotlib

    matplotlib.use('TkAgg')


def _get_vmin_and_vmax(data, diverging: bool = False, color_quantile: float = 0, **_):
    if not diverging:
        return None, None

    vmin = np.quantile(data, color_quantile)
    vmax = np.quantile(data, 1 - color_quantile)

    if abs(vmin) > vmax:  # only possible when vmin < 0
        return vmin, -vmin
    return -vmax, vmax


def _draw_color_bar(fig, ax, plot):
    try:
        norm = matplotlib.colors.Normalize(vmin=plot.cvalues.min(), vmax=plot.cvalues.max())
        plot = plt.cm.ScalarMappable(norm=norm, cmap=plot.cmap)
        plot.set_array([])

    except AttributeError:
        pass

    fig.colorbar(plot, cax=ax)
    ax.tick_params(labelsize=7, right=False, direction="in")


def _update_axis_projection(ax, axi, projection):
    rows, cols, start, stop = axi.get_subplotspec().get_geometry()
    ax.flat[start].remove()
    ax.flat[start] = plt.gcf().add_subplot(rows, cols, start + 1, projection=projection)


def _set_map_projection(ax, axi, projection=None,
                        coastlines: bool = False, coastline_kwargs: dict = {"linewidth": 0.5}, **_):
    if projection:
        _update_axis_projection(ax, axi, projection=projection)

        if coastlines:
            axi.coastlines(**coastline_kwargs)

    else:
        axi.tick_params(labelsize=9)
        axi.xaxis.set_major_formatter(FormatStrFormatter("%d°"))
        axi.yaxis.set_major_formatter(FormatStrFormatter("%d°"))


def _set_plot_kwargs(data: np.ndarray, plot_type: str, **kwargs):
    plot_kwargs = {}

    if (projection := kwargs.get("projection")) is None:
        plot_kwargs["aspect"] = "auto"
    elif isinstance(projection, projections.Projection):
        plot_kwargs["transform"] = projections.PlateCarree()
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
        _plot_latitude_longitude_contour(data, format_title(dataset.variable, time, lev), **kwargs)

    plt.show()


def _plot_latitude_longitude_contour(data: np.ndarray, title, figsize: tuple[int, int] = (8, 5),
                                     plot_type: str = "image", cbar: bool = True, **kwargs) -> None:
    """
    :param data:
    :param cmap:
    :param figsize:
    :param plot_type:
    :param cbar:

    :param diverging
    :param color_quantile

    :param projection
    :param coastlines
    :param coastline_kwargs

    :return:
    """

    fig, ax = plt.subplots(ncols=2, figsize=figsize, tight_layout=True, width_ratios=(97, 3))
    _set_map_projection(ax, ax[0], **kwargs)

    plot_kwargs = _set_plot_kwargs(data, plot_type, **kwargs)

    ax[0].set_title(title, fontsize=9)
    ax[0].grid(True, which="both", linestyle="dashed", linewidth=0.15)

    if plot_type.startswith("contour"):
        plot = ax[0].contourf if plot_type == "contourf" else ax[0].contour
        lats = np.linspace(-90, 90, data.shape[0])
        lons = np.linspace(-180, 180, data.shape[1])

        plotted_data = plot(*np.meshgrid(lons, lats), data, **plot_kwargs)

    elif plot_type == "image":
        plotted_data = ax[0].imshow(data, origin="lower", extent=[-180, 180, -90, 90], **plot_kwargs)

    else:
        raise ValueError(f"Unknown plot type {plot_type!r}")

    if cbar:
        _draw_color_bar(fig, ax[1], plotted_data)

    # plt.savefig("assets/contours/" + output + ".png", dpi=300)
