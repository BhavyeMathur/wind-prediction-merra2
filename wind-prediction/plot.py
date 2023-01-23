from typing import TypedDict, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter
import seaborn

from data_loading import *
from merra2 import *

seaborn.set_theme()

mpl.rc("font", family="serif", serif=["Verdana"])


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


def get_vmin_and_vmax(data):
    vmin = np.min(data)
    vmax = np.max(data)

    if vmin < 0 < vmax:
        if abs(vmin) > vmax:
            vmax = -vmin
        elif vmax > abs(vmin):
            vmin = -vmax

    return vmin, vmax


# Basic Plot Functions

def setup_plot(title: str,
               xlabel: str | None,
               ylabel: str | None,
               subtitle: str | None,
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
                               valmin=valmin2, valmax=valmax2, valstep=1, valinit=16,
                               color=(0.2, 0.2, 0.2))
    horizontal_slider.valtext.set_visible(False)
    horizontal_slider.label.set_visible(False)

    return fig, ax, color_bar_ax, vertical_slider, horizontal_slider


def create_1x2_plot(title: str, figsize: tuple[int, int] = (12, 5), **kwargs):
    fig, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, num=title, figsize=figsize, tight_layout=True, **kwargs)
    return fig, ax11, ax12


def create_2x2_plot(title: str, figsize: tuple[int, int] = (12, 5), **kwargs):
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(nrows=2, ncols=2, num=title, figsize=figsize, tight_layout=True,
                                                     **kwargs)
    return fig, ax11, ax12, ax21, ax22


# Drawing Methods

def load_map(shape):
    coastlines = mpimg.imread("assets/equirectangular_projection.png")[:, :, 0]
    coastline_latitudes = np.linspace(0, shape[0], coastlines.shape[0])
    coastline_longitudes = np.linspace(0, shape[1], coastlines.shape[1])

    return coastlines, coastline_latitudes, coastline_longitudes


def draw_map(ax, coastlines, latitudes, longitudes):
    ax.contour(longitudes, latitudes, coastlines, levels=1, colors=[(0, 0, 0, 0), (0, 0, 0, 0.2)])


def plot_variable_at_time_level_and_latitude_vs_longitude(filename: str,
                                                          variable: str,
                                                          time: int,
                                                          level: int,
                                                          latitude: int,
                                                          folder: str = "compressed"):
    data = load_variable_at_time_level_and_latitude(filename, variable, time, level, latitude, folder=folder)

    plt.plot(np.linspace(-180, 180, 576), data)

    plt.title(f"{variable} ({get_units_from_variable(variable)}) at {format_latitude(latitude)}, {format_level(level)}"
              f" on {format_date(filename)} at {format_time(time)}", fontsize=8)
    plt.show()


def plot_variable_at_time_level_and_longitude_vs_latitude(filename: str,
                                                          variable: str,
                                                          time: int,
                                                          level: int,
                                                          longitude: int,
                                                          folder: str = "compressed"):
    data = load_variable_at_time_level_and_longitude(filename, variable, time, level, longitude, folder=folder)

    plt.plot(np.linspace(-90, 90, 361), data)

    plt.title(f"{variable} ({get_units_from_variable(variable)}) at {format_longitude(longitude)}, "
              f"{format_level(level)} on {format_date(filename)} at {format_time(time)}", fontsize=8)
    plt.show()


def plot_variable_at_time_and_level(filename: str = "",
                                    variable: str = "",
                                    time: int = 0,
                                    level: int = 71,
                                    folder: str = "compressed",
                                    lat_start: int = 0,
                                    lat_end: int = 361,
                                    lat_step: int = 1,
                                    fig_ax1_ax2=None,
                                    data: np.ndarray = None,
                                    linewidth=0.2):
    if data is None:
        if isinstance(level, int):
            data = load_variable_at_time_and_level(filename, variable, time, level, folder=folder)
        else:
            data = load_variable_at_time_and_level(filename, variable, time, int(level), folder) * (1 - level % 1)
            data += load_variable_at_time_and_level(filename, variable, time, int(level) + 1, folder) * (level % 1)

    title = f"{variable} ({get_units_from_variable(variable)}) at {format_level(level)}" \
            f" on {format_date(filename)} at {format_time(time)}"

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


def plot_contour_at_time_and_level(filename: str,
                                   variable: str,
                                   time: int,
                                   level: int,
                                   folder: str = "compressed",
                                   data: np.ndarray | None = None,
                                   **kwargs) -> None:
    data = load_variable_at_time_and_level(filename, variable, time, level, folder=folder)

    plt.contourf(np.linspace(-180, 180, 576), np.linspace(-90, 90, 361), data, cmap="viridis", **kwargs)

    plt.title(f"{variable} ({get_units_from_variable(variable)}) at {format_level(level)}"
              f" on {format_date(filename)} at {format_time(time)}", fontsize=8)
    plt.show()


def plot_interactive_contour_at_time(filename: str,
                                     variable: str,
                                     time: int,
                                     title: str,
                                     folder: str = "compressed",
                                     latitude_samples: int = 180,
                                     longitude_samples: int | None = None):
    data = load_variable_at_time(filename, variable, time, folder=folder)
    data = interpolate_variable_at_time(data, latitude_samples, longitude_samples)

    fig, ax, color_bar_ax, level_slider, resolution_slider = create_2_interactive_sliders_with_color_bar(
        title, 0, 71, 2, 32)

    def update(_):
        color_bar_ax.clear()
        ax.clear()
        ax.set_yticks([])
        ax.set_xticks([])

        subdata = data[71 - level_slider.val]
        vmin, vmax = get_vmin_and_vmax(subdata)

        contour = contourf(ax, resolution_slider.val, subdata, vmin=vmin, vmax=vmax)
        fig.colorbar(contour, cax=color_bar_ax, fraction=0.05, pad=0.02)

    level_slider.on_changed(update)
    resolution_slider.on_changed(update)

    update(0)
    plt.show()
