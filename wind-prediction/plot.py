from typing import TypedDict, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter
import seaborn
import cmasher as cmr

from data_loading import *
from merra2 import *

seaborn.set_theme()


class DataDictionary(TypedDict, total=False):
    """Class representing the data format for the plotting functions
    """
    data: Sequence
    label: str | None
    size: float | None
    colour: tuple[int, int, int] | Sequence[tuple[int, int, int]] | None


# Color & Gradient Functions


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Converts an RGB colour to a Hex string

    @param r: red value of the colour (0-255)
    @param g: green value of the colour (0-255)
    @param b: blue value of the colour (0-255)
    @return: a hex string representing the input colour
    """
    return "#%02x%02x%02x" % (r, g, b)


def create_gradient(colour1: tuple[int, int, int],
                    colour2: tuple[int, int, int]) -> ListedColormap:
    """Creates a linear colour gradient between 2 colours

    @param colour1: the start colour as an RGB tuple
    @param colour2: the end colour as an RGB tuple
    @return: a matplotlib ListedColormap gradient
    """
    vals: np.ndarray = np.ones((256, 4))  # creates a list of 256 tuples of size 4 (RGBA) with 1s for Alpha
    vals[:, 0] = np.linspace(colour1[0] / 255, colour2[0] / 255, 256)
    vals[:, 1] = np.linspace(colour1[1] / 255, colour2[1] / 255, 256)
    vals[:, 2] = np.linspace(colour1[2] / 255, colour2[2] / 255, 256)

    return ListedColormap(vals)


def combine_gradients(gradient1: ListedColormap,
                      gradient2: ListedColormap,
                      prop: float = 0.5) -> ListedColormap:
    """Combines 2 colour gradients into 1

    @param gradient1: the first colour gradient
    @param gradient2: the second colour gradient
    @param prop: proportion representing the influence of the first gradient

    @return: a matplotlib ListedColormap gradient
    """

    return ListedColormap(np.vstack((gradient1(np.linspace(0, 1, int(prop * 256))),
                                     gradient2(np.linspace(0, 1, 256 - int(prop * 256))))))


PRIMARY_DARK = (111, 194, 75)
PRIMARY = (150, 213, 87)
PRIMARY_LIGHT = (189, 226, 92)

SECONDARY_DARK = (192, 61, 63)
SECONDARY = (220, 61, 70)
SECONDARY_LIGHT = (213, 87, 89)

ACCENT_DARK = (255, 182, 0)
ACCENT = (246, 212, 61)

gradient = create_gradient(colour1=PRIMARY_DARK,
                           colour2=PRIMARY_LIGHT)
colour_order = [PRIMARY, SECONDARY, ACCENT, (78, 140, 217)]


def _get_colour(i: int):
    if i < len(colour_order):
        return rgb_to_hex(*colour_order[i])
    else:
        return None


def get_vmin_and_vmax(data, quantile: float = 0):
    vmin = np.quantile(data, quantile)
    vmax = np.quantile(data, 1 - quantile)

    if vmin < 0 < vmax:
        if abs(vmin) > vmax:
            vmax = -vmin
        elif vmax > abs(vmin):
            vmin = -vmax

    if vmin >= 0:
        vmin = -vmax

    if vmax <= 0:
        vmax = -vmin

    return vmin, vmax


# Basic Plot Functions

def setup_plot(title: str,
               xlabel: str | None,
               ylabel: str | None,
               subtitle: str | None = None,
               xlabel_rotation: int = 0,
               legend: bool = True) -> None:
    """Adds the title, labels, and other elements to the plot and then shows/saves it

    For internal use by the other plotting functions

    @param title: the title of the graph
    @param subtitle: the optional subtitle of the graph
    @param xlabel_rotation: the rotation (degrees) of the x labels
    @param legend: show the legend?

    @param xlabel: the label of the x-axis
    @param ylabel: the label of the y-axis
    """
    if subtitle is None:
        plt.title(title, fontsize=8)
    else:
        plt.suptitle(title, fontsize=15)
        plt.title(subtitle, pad="15.0")

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.xticks(rotation=xlabel_rotation)

    if legend:
        plt.legend()

    plt.show()


def plot_histogram(data: list[DataDictionary],
                   title: str,
                   xlabel: str | None = None,
                   ylabel: str | None = None,
                   subtitle: str | None = None,
                   bins: int | None = None) -> None:
    """Plots a histogram with the data supplied

    @param data: a dictionary represent the data, labels, and colours for the histogram

    @param title: the title of the graph
    @param subtitle: the optional subtitle of the graph

    @param xlabel: the label of the x-axis
    @param ylabel: the label of the y-axis
    @param bins: the number of bins to create in the histogram
    """

    series_list = []
    labels = []
    colours = []

    for i, series in enumerate(data):
        series_list.append(series["data"])
        labels.append(series.get("label", ""))
        colours.append(series.get("colour", _get_colour(i)))

    n, bins, patches = plt.hist(series_list, bins=bins, label=labels, color=colours, linewidth=0)

    if len(series_list) == 1:
        for i in range(len(patches)):
            patches[i].set_facecolor(gradient(n[i] / max(n)))

    setup_plot(title=title, xlabel=xlabel, ylabel=ylabel, subtitle=subtitle)


# Plotting templates

def create_interactive_slider_with_color_bar(title: str,
                                             valmin1: float,
                                             valmax1: float,
                                             width_ratios: tuple[int, int, int] = (2, 98, 3),
                                             figsize: tuple[int, int] = (8, 5),
                                             **kwargs):
    fig, (level_slider_axis, ax, color_bar_ax), \
        = plt.subplots(nrows=1,
                       ncols=3,
                       num=title,
                       figsize=figsize,
                       width_ratios=width_ratios,
                       tight_layout=True,
                       gridspec_kw={"left": 0.025, "right": 0.9, "bottom": 0.045, "top": 0.95, "hspace": 0.1,
                                    "wspace": 0.1},
                       **kwargs)

    vertical_slider = Slider(ax=level_slider_axis, label="", orientation="vertical",
                             valmin=valmin1, valmax=valmax1, valinit=0, valstep=1,
                             color=(0.2, 0.2, 0.2))
    vertical_slider.valtext.set_visible(False)
    vertical_slider.label.set_visible(False)

    return fig, ax, color_bar_ax, vertical_slider


def create_2_interactive_sliders_with_color_bar(title: str,
                                                valmin1: float,
                                                valmax1: float,
                                                valmin2: float,
                                                valmax2: float,
                                                valinit2: float,
                                                width_ratios: tuple[int, int, int] = (2, 98, 3),
                                                height_ratios: tuple[int, int] = (98, 2),
                                                figsize=(8, 5),
                                                **kwargs):
    fig, ((level_slider_axis, ax, color_bar_ax), (corner1, resolution_slider_axis, corner2)) \
        = plt.subplots(nrows=2,
                       ncols=3,
                       num=title,
                       figsize=figsize,
                       width_ratios=width_ratios,
                       height_ratios=height_ratios,
                       tight_layout=True,
                       gridspec_kw={"left": 0.025, "right": 0.9, "bottom": 0.045, "top": 0.95, "hspace": 0.1,
                                    "wspace": 0.1},
                       **kwargs)

    corner1.set_visible(False)
    corner2.set_visible(False)

    vertical_slider = Slider(ax=level_slider_axis, label="", orientation="vertical",
                             valmin=valmin1, valmax=valmax1, valstep=1, valinit=0,
                             color=(0.2, 0.2, 0.2))
    vertical_slider.valtext.set_visible(False)
    vertical_slider.label.set_visible(False)

    horizontal_slider = Slider(ax=resolution_slider_axis, label="",
                               valmin=valmin2, valmax=valmax2, valinit=valinit2,
                               color=(0.2, 0.2, 0.2))
    horizontal_slider.valtext.set_visible(False)
    horizontal_slider.label.set_visible(False)

    return fig, ax, color_bar_ax, vertical_slider, horizontal_slider


def create_1x1_plot(title: str, figsize: tuple[int, int] = (8, 5), **kwargs):
    return plt.subplots(nrows=1, ncols=1, num=title, figsize=figsize, tight_layout=True, **kwargs)


def create_1x2_plot(title: str, figsize: tuple[int, int] = (12, 5), **kwargs):
    fig, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, num=title, figsize=figsize, tight_layout=True, **kwargs)
    return fig, ax11, ax12


def create_1x3_plot(title: str,
                    figsize: tuple[int, int] = (12, 5),
                    width_ratios: tuple[int, int, int] = (50, 50, 3),
                    **kwargs):
    fig, (ax11, ax12, ax13) = plt.subplots(nrows=1, ncols=3, num=title, figsize=figsize, tight_layout=True,
                                           width_ratios=width_ratios, **kwargs)
    return fig, ax11, ax12, ax13


def create_4x3_plot(title: str,
                    figsize: tuple[int, int] = (11.69, 8.27),
                    **kwargs):
    return plt.subplots(nrows=4, ncols=3, num=title, figsize=figsize, sharex=True, sharey=True,
                        **kwargs)


def create_4x4_plot(title: str,
                    figsize: tuple[int, int] = (11.69, 8.27),
                    **kwargs):
    return plt.subplots(nrows=4, ncols=4, num=title, figsize=figsize, sharex=True, sharey=True,
                        **kwargs)


def create_4x6_plot(title: str,
                    figsize: tuple[int, int] = (11.69, 8.27),
                    **kwargs):
    return plt.subplots(nrows=4, ncols=6, num=title, figsize=figsize, sharex=True, sharey=True,
                        **kwargs)


def create_6x6_plot(title: str,
                    figsize: tuple[int, int] = (11.69, 8.27),
                    **kwargs):
    return plt.subplots(nrows=6, ncols=6, num=title, figsize=figsize, sharex=True, sharey=True, **kwargs)


def create_8x6_plot(title: str,
                    figsize: tuple[int, int] = (11.69, 8.27),
                    **kwargs):
    return plt.subplots(nrows=8, ncols=6, num=title, figsize=figsize, sharex=True, sharey=True, **kwargs)


def create_2x2_plot(title: str, figsize: tuple[int, int] = (12, 5), **kwargs):
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(nrows=2, ncols=2, num=title, figsize=figsize, tight_layout=True,
                                                     **kwargs)
    return fig, ax11, ax12, ax21, ax22


# Drawing Methods

def load_map(shape):
    coastlines = mpimg.imread("assets/equirectangular_projection.png")[::-1, :, 0]
    coastline_latitudes = np.linspace(-90, 90, coastlines.shape[0])
    coastline_longitudes = np.linspace(-180, 180, coastlines.shape[1])

    return coastlines, coastline_latitudes, coastline_longitudes


def draw_map(ax, coastlines, latitudes, longitudes):
    ax.contourf(longitudes, latitudes, coastlines, levels=1, colors=[(0, 0, 0, 0), (0, 0, 0, 0.2)])


def plot_variable_at_time_level_and_latitude_vs_longitude(filename: str,
                                                          variable: str,
                                                          time: int,
                                                          level: int | float,
                                                          latitude: int,
                                                          data: np.ndarray = None,
                                                          **kwargs):
    if data is None:
        data = load_variable_at_time_level_and_latitude(filename, variable, time, level, latitude, **kwargs)

    plt.plot(np.linspace(-180, 180, 576), data)

    plt.title(f"{variable} ({get_units_from_variable(variable)}) at {format_latitude(latitude)}, {format_level(level)}"
              f" on {format_date(filename)} at {format_time(time, filename)}", fontsize=8)
    plt.show()


def plot_yavg_variable_at_time_level_and_latitude_vs_longitude(filename: str,
                                                               variable: str,
                                                               time: int,
                                                               level: int | float,
                                                               latitude: int,
                                                               years: tuple[int, ...] = (1980, 1981, 1990, 1991,
                                                                                         2000, 2001, 2010, 2011)):
    avg = np.zeros((576,), dtype="float64")
    for i, year in enumerate(years):
        data = load_variable_at_time_level_and_latitude(filename.format(get_merra_stream_from_year(year), year),
                                                        variable, time, level, latitude)
        avg += data
        plt.plot(np.linspace(-180, 180, 576), data, label=year,
                 linewidth=0.5, color=plt.cm.tab20b(i / len(years)), alpha=0.5)

    avg /= len(years)
    plt.plot(np.linspace(-180, 180, 576), avg, linestyle="dashed", label="Average", color="#fff")

    plt.title(f"{variable} ({get_units_from_variable(variable)}) at {format_latitude(latitude)}, {format_level(level)}"
              f" at {format_time(time, filename)}", fontsize=8)
    plt.show()


def plot_variable_at_time_level_and_longitude_vs_latitude(filename: str,
                                                          variable: str,
                                                          time: int,
                                                          level: int | float,
                                                          longitude: int,
                                                          data: np.ndarray = None):
    if data is None:
        data = load_variable_at_time_level_and_longitude(filename, variable, time, level, longitude)

    plt.plot(np.linspace(-90, 90, 361), data)

    plt.title(f"{variable} ({get_units_from_variable(variable)}) at {format_longitude(longitude)}, "
              f"{format_level(level)} on {format_date(filename)} at {format_time(time, filename)}", fontsize=8)
    plt.show()


def plot_variable_at_time_and_level_vs_longitude(filename: str,
                                                 variable: str,
                                                 time: int,
                                                 level: int | float,
                                                 lat_start: int = 0,
                                                 lat_end: int = 361,
                                                 lat_step: int = 1,
                                                 fig_ax1_ax2=None,
                                                 data: np.ndarray = None,
                                                 linewidth=0.2):
    if data is None:
        data = load_variable_at_time_and_level(filename, variable, time, level)

    title = f"{variable} ({get_units_from_variable(variable)}) at {format_level(level)}" \
            f" on {format_date(filename)} at {format_time(time, filename)}"

    if fig_ax1_ax2:
        fig, ax1, ax2 = fig_ax1_ax2
    else:
        fig, ax1, ax2 = create_1x2_plot(title, figsize=(8, 5), width_ratios=(98, 2))
        plt.show()

    for latitude in range(lat_start, lat_end, lat_step):
        ax1.plot(np.linspace(-180, 180, 576), data[latitude],
                 color=plt.cm.coolwarm(latitude / 361),
                 linewidth=linewidth)

    bar = mpl.colorbar.Colorbar(ax2, cmap="coolwarm", orientation="vertical", values=np.linspace(-90, 90, 50))
    bar.set_ticks([-90, -60, -30, 0, 30, 60, 90])
    bar.set_ticklabels(["-90°", "-60°", "-30°", " 0°", "30°", "60°", "90°"])

    ax1.set_xlim((-180, 180))
    ax1.tick_params(labelsize=9)
    ax1.set_title(title, fontsize=9)
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%d°"))

    ax2.tick_params(labelsize=7, right=False, direction="in")

    return fig, ax1, ax2


def _plot_variable_at_time_and_longitude_vs_vertical(filename: str,
                                                     variable: str,
                                                     time: int,
                                                     longitude: int,
                                                     verticals: list[float] | np.ndarray,
                                                     lat_start: int = 0,
                                                     lat_end: int = 361,
                                                     lat_step: int = 1,
                                                     data: np.ndarray = None,
                                                     fig_ax1_ax2=None,
                                                     linewidth=0.2):
    if data is None:
        data = load_variable_at_time_and_longitude(filename, variable, time, longitude)

    title = f"{format_variable(variable)} at {format_longitude(longitude)}" \
            f" on {format_date(filename)} at {format_time(time, filename)}"

    if fig_ax1_ax2:
        fig, ax1, ax2 = fig_ax1_ax2
    else:
        fig, ax1, ax2 = create_1x2_plot(title, figsize=(8, 5), width_ratios=(98, 2))
        plt.show()

    for latitude in range(lat_start, lat_end, lat_step):
        ax1.plot(verticals[-data.shape[0]:], data[:, latitude],
                 color=plt.cm.coolwarm(latitude / 361),
                 linewidth=linewidth)

    bar = mpl.colorbar.Colorbar(ax2, cmap="coolwarm", orientation="vertical", values=np.linspace(-90, 90, 50))
    bar.set_ticks([-90, -60, -30, 0, 30, 60, 90])
    bar.set_ticklabels(["-90°", "-60°", "-30°", " 0°", "30°", "60°", "90°"])

    ax1.tick_params(labelsize=9)
    ax1.set_title(title, fontsize=9)

    ax2.tick_params(labelsize=7, right=False, direction="in")

    return fig, ax1, ax2


def _plot_variable_at_time_and_latitude_vs_vertical(filename: str,
                                                    variable: str,
                                                    time: int,
                                                    latitude: int,
                                                    verticals: list[float] | np.ndarray,
                                                    lon_start: int = 0,
                                                    lon_end: int = 576,
                                                    lon_step: int = 1,
                                                    data: np.ndarray = None,
                                                    fig_ax1_ax2=None,
                                                    linewidth=0.2):
    if data is None:
        data = load_variable_at_time_and_latitude(filename, variable, time, latitude)

    title = f"{variable} ({get_units_from_variable(variable)}) at {format_latitude(latitude)}" \
            f" on {format_date(filename)} at {format_time(time, filename)}"

    if fig_ax1_ax2:
        fig, ax1, ax2 = fig_ax1_ax2
    else:
        fig, ax1, ax2 = create_1x2_plot(title, figsize=(8, 5), width_ratios=(98, 2))
        plt.show()

    for longitude in range(lon_start, lon_end, lon_step):
        ax1.plot(verticals[-data.shape[0]:], data[:, longitude],
                 color=plt.cm.coolwarm(longitude / 576),
                 linewidth=linewidth)

    bar = mpl.colorbar.Colorbar(ax2, cmap="coolwarm", orientation="vertical", values=np.linspace(-180, 180, 50))
    bar.set_ticks([-180, -120, -60, 0, 60, 120, 180])
    bar.set_ticklabels(["-180°", "-120°", "-60°", " 0°", "60°", "120°", "180°"])

    ax1.tick_params(labelsize=9)
    ax1.set_title(title, fontsize=9)

    ax2.tick_params(labelsize=7, right=False, direction="in")

    return fig, ax1, ax2


def plot_variable_at_level_and_longitude_vs_time(filename: str,
                                                 variable: str,
                                                 level: int,
                                                 longitude: int,
                                                 fig_ax1_ax2=None,
                                                 linewidth=0.2):
    data = load_variable_at_level_and_longitude(filename, variable, longitude, level)

    title = f"{format_variable(variable)} at {format_longitude(longitude)} at {format_level(level)}"

    if fig_ax1_ax2:
        fig, ax1, ax2 = fig_ax1_ax2
    else:
        fig, ax1, ax2 = create_1x2_plot(title, figsize=(8, 5), width_ratios=(98, 2))
        plt.show()

    for latitude in range(361):
        ax1.plot(data[:, latitude],
                 color=plt.cm.coolwarm(latitude / 361),
                 linewidth=linewidth)

    bar = mpl.colorbar.Colorbar(ax2, cmap="coolwarm", orientation="vertical", values=np.linspace(-90, 90, 50))
    bar.set_ticks([-90, -60, -30, 0, 30, 60, 90])
    bar.set_ticklabels(["-90°", "-60°", "-30°", " 0°", "30°", "60°", "90°"])

    ax1.tick_params(labelsize=9)
    ax1.set_title(title, fontsize=9)

    ax2.tick_params(labelsize=7, right=False, direction="in")

    return fig, ax1, ax2


def plot_variable_at_time_and_longitude_vs_level(filename: str,
                                                 variable: str,
                                                 time: int,
                                                 longitude: int,
                                                 **kwargs):
    output = f"{variable}/{format_longitude(longitude, for_output=True)}" \
             f"-{format_date(filename, for_output=True)}-{format_time(time, filename)}-vs-level"

    values = _plot_variable_at_time_and_longitude_vs_vertical(filename, variable, time, longitude, np.arange(0, 72),
                                                              **kwargs)

    plt.savefig("assets/vs-vertical/" + output + ".png", dpi=300)
    return values


def plot_variable_at_time_and_longitude_vs_pressure(filename: str,
                                                    variable: str,
                                                    time: int,
                                                    longitude: int,
                                                    **kwargs):
    output = f"{variable}/{format_longitude(longitude, for_output=True)}" \
             f"-{format_date(filename, for_output=True)}-{format_time(time, filename)}-vs-pressure"

    fig, ax1, ax2 = _plot_variable_at_time_and_longitude_vs_vertical(filename, variable, time, longitude,
                                                                     [get_pressure_from_level(lev)
                                                                      for lev in range(72)], **kwargs)
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%d hPa"))

    plt.savefig("assets/vs-vertical/" + output + ".png", dpi=300)
    return fig, ax1, ax2


def plot_variable_at_time_and_longitude_vs_height(filename: str,
                                                  variable: str,
                                                  time: int,
                                                  longitude: int,
                                                  **kwargs):
    output = f"{variable}/{format_longitude(longitude, for_output=True)}" \
             f"-{format_date(filename, for_output=True)}-{format_time(time, filename)}-vs-height"

    heights = [height_from_pressure(get_pressure_from_level(lev) * 100) / 1000 for lev in range(72)]
    fig, ax1, ax2 = _plot_variable_at_time_and_longitude_vs_vertical(filename, variable, time, longitude, heights,
                                                                     **kwargs)
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%d km"))

    plt.savefig("assets/vs-vertical/" + output + ".png", dpi=300)
    return fig, ax1, ax2


def plot_variable_at_time_and_latitude_vs_level(filename: str,
                                                variable: str,
                                                time: int,
                                                latitude: int,
                                                **kwargs):
    output = f"{variable}/{format_latitude(latitude, for_output=True)}" \
             f"-{format_date(filename, for_output=True)}-{format_time(time, filename)}-vs-level"

    values = _plot_variable_at_time_and_latitude_vs_vertical(filename, variable, time, latitude, np.arange(0, 72),
                                                             **kwargs)
    plt.savefig("assets/vs-vertical/" + output + ".png", dpi=300)
    return values


def plot_variable_at_time_and_latitude_vs_pressure(filename: str,
                                                   variable: str,
                                                   time: int,
                                                   latitude: int,
                                                   **kwargs):
    output = f"{variable}/{format_latitude(latitude, for_output=True)}" \
             f"-{format_date(filename, for_output=True)}-{format_time(time, filename)}-vs-pressure"

    fig, ax1, ax2 = _plot_variable_at_time_and_latitude_vs_vertical(filename, variable, time, latitude,
                                                                    [get_pressure_from_level(lev)
                                                                     for lev in range(72)], **kwargs)
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%d hPa"))

    plt.savefig("assets/vs-vertical/" + output + ".png", dpi=300)
    return fig, ax1, ax2


def plot_variable_at_time_and_latitude_vs_height(filename: str,
                                                 variable: str,
                                                 time: int,
                                                 latitude: int,
                                                 **kwargs):
    output = f"{variable}/{format_latitude(latitude, for_output=True)}" \
             f"-{format_date(filename, for_output=True)}-{format_time(time, filename)}-vs-height"

    heights = [height_from_pressure(get_pressure_from_level(lev) * 100) / 1000 for lev in range(72)]
    fig, ax1, ax2 = _plot_variable_at_time_and_latitude_vs_vertical(filename, variable, time, latitude, heights,
                                                                    **kwargs)
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%d km"))

    plt.savefig("assets/vs-vertical/" + output + ".png", dpi=300)
    return fig, ax1, ax2


# noinspection PyPep8Naming
def plot_3D_variable_at_time_and_level(filename: str,
                                       variable: str,
                                       time: int,
                                       level: int | float,
                                       ax=None,
                                       data: np.ndarray = None,
                                       linewidth: float = 0,
                                       alpha: float = 0.5,
                                       elevation: float = 30,
                                       azimuth: float = -130):
    if data is None:
        data = load_variable_at_time_and_level(filename, variable, time, level)

    title = f"{variable} ({get_units_from_variable(variable)}) at {format_level(level)}" \
            f" on {format_date(filename)} at {format_time(time, filename)}"

    if ax is None:
        ax = plt.axes(projection="3d")
        plt.show()

    lons, lats = np.meshgrid(np.linspace(-180, 180, 576), np.linspace(-90, 90, 361))
    ax.plot_surface(lons, lats, data, antialiased=True, cmap="coolwarm", linewidth=linewidth, alpha=alpha)

    ax.elev = elevation
    ax.azim = azimuth

    ax.set_xlim((-180, 180))
    ax.set_ylim((-90, 90))
    ax.tick_params(labelsize=9, color=(1, 1, 1, 0.1))
    ax.set_title(title, fontsize=9)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d°"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d°"))
    ax.zaxis.set_ticks([])

    ax.xaxis.set_pane_color((0, 0, 0, 0))
    ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))
    ax.xaxis.line.set_color((0, 0, 0, 0))
    ax.yaxis.line.set_color((0, 0, 0, 0))
    ax.zaxis.line.set_color((0, 0, 0, 0))


def plot_contour_at_time_and_level(filename: str,
                                   variable: str,
                                   time: int,
                                   level: int | float,
                                   data: np.ndarray | None = None,
                                   show_map: bool = False,
                                   data_processing: callable = lambda data: data,
                                   cmap="viridis",
                                   diverging: bool = False,
                                   **kwargs) -> None:
    if data is None:
        data = load_variable_at_time_and_level(filename, variable, time, level, **kwargs)
    data = data_processing(data)

    if get_nc4_dimensions(filename, **kwargs) == 3:
        output = f"{variable}" \
                 f"-{format_date(filename, for_output=True)}-{format_time(time, filename)}"
        title = f"{format_variable(variable)} on {format_date(filename)} at {format_time(time, filename)}"
    else:
        output = f"{variable}-{format_level(level, for_output=True)}" \
                 f"-{format_date(filename, for_output=True)}-{format_time(time, filename)}"
        title = f"{format_variable(variable)} at {format_level(level)}" \
                f" on {format_date(filename)} at {format_time(time, filename)}"

    fig, ax1, ax2 = create_1x2_plot(title, figsize=(8, 5), width_ratios=(98, 2))

    if diverging:
        vmin, vmax = get_vmin_and_vmax(data)
    else:
        vmin, vmax = None, None

    contour = ax1.imshow(data, cmap=cmap, origin="lower", aspect="auto",
                         vmin=vmin, vmax=vmax, extent=[-180, 180, -90, 90])
    fig.colorbar(contour, cax=ax2, fraction=0.05, pad=0.02)

    if show_map:
        output += "-map"
        draw_map(ax1, *load_map(data.shape))

    ax1.tick_params(labelsize=9)
    ax1.set_title(title, fontsize=9)
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%d°"))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%d°"))
    ax1.set_xlim((-180, 180))
    ax1.set_ylim((-90, 90))

    ax2.tick_params(labelsize=7, right=False, direction="in")

    plt.savefig("assets/contours/" + output + ".png", dpi=300)


def plot_contour_grid_at_level(filename: str,
                               variable: str,
                               level: int | float,
                               data_processing: callable = lambda data: data,
                               cmap="viridis",
                               diverging: bool = False,
                               **kwargs) -> None:
    if get_nc4_dimensions(filename.format(1), **kwargs) == 3:
        output = f"{variable}-{get_year_from_filename(filename)}"
        title = f"{format_variable(variable)} during {get_year_from_filename(filename)}"
    else:
        output = f"{variable}-{format_level(level, for_output=True)}-{get_year_from_filename(filename)}"
        title = f"{format_variable(variable)} during {get_year_from_filename(filename)} at {format_level(level)}"

    fig, axes = create_6x6_plot(title)

    plt.subplots_adjust(left=0.03, right=0.97,
                        bottom=0.15, top=0.85,
                        wspace=0.01, hspace=0.01)
    fig.suptitle(title, fontsize=12, y=0.95)

    axes[0][0].set_xticks([])
    axes[0][0].set_yticks([])

    hours = get_nc4_dimension_size(filename.format(1), "time", **kwargs)

    if hours == 24:
        axes[0][0].set_ylabel("00:30", fontsize=9)
        axes[1][0].set_ylabel("04:30", fontsize=9)
        axes[2][0].set_ylabel("08:30", fontsize=9)
        axes[3][0].set_ylabel("12:30", fontsize=9)
        axes[4][0].set_ylabel("16:30", fontsize=9)
        axes[5][0].set_ylabel("20:30", fontsize=9)
        hours = [0, 4, 8, 12, 16, 20]
    else:
        axes[0][0].set_ylabel("01:30", fontsize=9)
        axes[1][0].set_ylabel("04:30", fontsize=9)
        axes[2][0].set_ylabel("10:30", fontsize=9)
        axes[3][0].set_ylabel("13:30", fontsize=9)
        axes[4][0].set_ylabel("19:30", fontsize=9)
        axes[5][0].set_ylabel("22:30", fontsize=9)
        hours = [0, 1, 3, 4, 6, 7]

    for month in range(1, 13, 2):

        x_index = (month - 1) // 2
        axes[0][x_index].set_title(format_month(month - 1), fontsize=9)

        for i, hour in enumerate(hours):
            current_file = filename.format(month)
            data = load_variable_at_time_and_level(current_file, variable, hour, level, **kwargs)
            data = data_processing(data)

            if get_nc4_dimensions(current_file, **kwargs) == 3:
                title = f"{format_variable(variable)} on {format_date(current_file)} " \
                        f"at {format_time(hour, current_file)}"
            else:
                title = f"{format_variable(variable)} at {format_level(level)}" \
                        f" on {format_date(current_file)} at {format_time(hour, current_file)}"

            if diverging:
                vmin, vmax = get_vmin_and_vmax(data)
            else:
                vmin, vmax = None, None

            axes[i][x_index].imshow(data, cmap=cmap, origin="lower", aspect="auto",
                                    vmin=vmin, vmax=vmax, extent=[-180, 180, -90, 90])

    plt.savefig("assets/contours/" + output + ".png", dpi=300)


def plot_interactive_contour_at_time(filename: str,
                                     variable: str,
                                     time: int,
                                     samples: int = 150,
                                     data: np.ndarray | None = None):
    if data is None:
        data = load_variable_at_time(filename, variable, time)
    data = interpolate_variable_at_time(data, samples, samples * 2)

    title = f"{get_variable_name_from_code(variable)} ({get_units_from_variable(variable)})"
    fig, ax, color_bar_ax, level_slider = create_interactive_slider_with_color_bar(title, 0, data.shape[0] - 1)
    ax.set_yticks([])
    ax.set_xticks([])

    def update_contour(val):
        ax.clear()
        ax.set_yticks([])
        ax.set_xticks([])

        color_bar_ax.clear()

        contour = ax.imshow(data[data.shape[0] - 1 - val], cmap="viridis", aspect="auto", origin="lower")
        fig.colorbar(contour, cax=color_bar_ax, fraction=0.05, pad=0.02)

    level_slider.on_changed(update_contour)

    update_contour(0)
    plt.show()

    return level_slider


def plot_ppc_by_varying_argument(function: callable,
                                 arg_name: str,
                                 arg_min: float,
                                 arg_max: float,
                                 title: str,
                                 xlabel: str,
                                 ylabel: str,
                                 n: int = 100,
                                 *args, **kwargs):
    x = []
    y = []

    for arg in np.linspace(arg_min, arg_max, n):
        x_val, y_val = function(*args, **{arg_name: arg}, **kwargs)
        x.append(x_val)
        y.append(y_val)

    plt.plot(x, y)
    setup_plot(title, xlabel, ylabel)
