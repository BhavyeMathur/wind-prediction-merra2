import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import FuncAnimation
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


def _draw_color_bar(fig: plt.Figure, data, cmap=cmr.ocean, cbar: bool = True, **kwargs) -> None:
    if not cbar:
        return

    vmin, vmax = _get_vmin_and_vmax(data, **kwargs)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    plot = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
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


def _new_earth_figure(title: str, output: list[str], projection=_PLATE_CARREE,
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


# Source: Eelco van Vliet
# https://stackoverflow.com/questions/16915966/using-matplotlib-animate-to-animate-a-contour-plot-in-python
def _clean_up_artists(axis, artist_list) -> None:
    """
    try to remove the artists stored in the artist list belonging to the 'axis'.

    :param axis: clean artists belonging to these axis
    :param artist_list: list of artist to remove
    """
    for artist in artist_list:
        try:
            # fist attempt: try to remove collection of contours for instance
            while artist.collections:
                for col in artist.collections:
                    artist.collections.remove(col)
                    try:
                        axis.collections.remove(col)
                    except ValueError:
                        pass

                artist.collections = []
                axis.collections = []
        except AttributeError:
            pass

        # second attempt, try to remove the text
        try:
            artist.remove()
        except (AttributeError, ValueError):
            pass


def plot_dataset(dataset: MERRA2Dataset, time: None | str | list[str] = None, lev: None | int = None,
                 latitude: None | int = None, data_transform=lambda data: data, **kwargs) -> None:
    data = dataset.load(time=time, lev=lev, lat=latitude)
    data = data_transform(data)

    # Latitude vs Longitude contour
    if len(data.shape) == 2:
        title = format_title(dataset.variable, time, lev)
        output = format_output(dataset.variable, time, lev)
        output = _plot_latitude_longitude_contour(data, title, output, **kwargs)

        plt.savefig(f"generated/contours/{'-'.join(output)}.png", dpi=300)
        plt.show()

    # Latitude vs Longitude contour, animated vs Time
    elif len(data.shape) == 3:
        if isinstance(time, str):
            times = [f"{hour:02}:30 {time}" for hour in range(1, 24, 3)]
            output = format_output(dataset.variable, times, lev)
        else:  # time is list
            times = [f"{hour:02}:30 {date}" for date in time for hour in range(1, 24, 3)]
            output = format_output(dataset.variable, time, lev)

        title = [format_title(dataset.variable, time, lev) for time in times]

        anim, output = _animate_latitude_longitude_contour_vs_time(data, title, output, times, **kwargs)
        anim.save(f"generated/contours/animated/{'-'.join(output)}.mp4", fps=15, dpi=300)

    else:
        raise ValueError(f"Data shape {data.shape} incompatible with plot")


def _animate_latitude_longitude_contour_vs_time(data: np.ndarray, title, output: list[str], times: list[str],
                                                nightshade: bool = False,  plot_type: str | list[str] = "image",
                                                **kwargs):
    fig, ax, mesh, plot_kwargs = _setup_latitude_longitude_contour(data, "", output, plot_type, **kwargs)
    plotted_data = []

    title_size = kwargs.get("title_size", 9)

    if nightshade:
        output.append("nightshade")

    def update(i):
        print(f"{i + 1}/{len(data)}")
        _clean_up_artists(ax, plotted_data)

        ax.set_title(title[i], fontsize=title_size)

        if nightshade:
            dt = datetime(*parse_datetime(times[i], 1980)[::-1])
            shade = ax.add_feature(Nightshade(dt, alpha=0.2))
            plotted_data.append(shade)

        plotted_data.append(_update_latitude_longitude_contour(mesh, data[i], ax, plot_type, **plot_kwargs))

        return plotted_data

    anim = FuncAnimation(fig, update, frames=tuple(range(len(data))), interval=0)

    return anim, output


def _plot_latitude_longitude_contour(data: np.ndarray, title, output: list[str],
                                     plot_type: str | list[str] = "image", **kwargs) -> list[str]:
    fig, ax, mesh, plot_kwargs = _setup_latitude_longitude_contour(data, title, output, plot_type, **kwargs)

    _update_latitude_longitude_contour(mesh, data, ax, plot_type, **plot_kwargs)

    return output


def _setup_latitude_longitude_contour(data: np.ndarray, title, output: list[str],
                                      plot_type: str | list[str] = "image", **kwargs):
    fig, ax = _new_earth_figure(title, output, **kwargs)
    plot_kwargs = _set_plot_kwargs(data, plot_type, **kwargs)
    mesh = _get_latlon_mesh(data.shape)

    if plot_type != "image":
        output.append(plot_type)

    _draw_color_bar(fig, data, **kwargs)

    return fig, ax, mesh, plot_kwargs


def _update_latitude_longitude_contour(mesh, data, ax, plot_type: str, **kwargs):
    if plot_type == "contourf":
        return ax.contourf(*mesh, data, **kwargs)

    elif plot_type == "contour":
        return ax.contour(*mesh, data, **kwargs)

    elif plot_type == "image":
        return ax.imshow(data, origin="lower", extent=(-180, 180, -90, 90), **kwargs)

    else:
        raise ValueError(f"Unknown plot type {plot_type!r}")
