import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from matplotlib import rc
import seaborn

from data_loading import *
from merra2 import *

seaborn.set_theme()

rc("font", family="serif", serif=["Verdana"])


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
                      prop: int | float = 0.5) -> ListedColormap:
    """Combines 2 colour gradients into 1

    @param gradient1: the first colour gradient
    @param gradient2: the second colour gradient
    @param prop: proportion representing the influence of the first gradient

    @return: a matplotlib ListedColormap gradient
    """

    return ListedColormap(np.vstack((gradient1(np.linspace(0, 1, int(prop * 256))),
                                     gradient2(np.linspace(0, 1, 256 - int(prop * 256))))))


def create_2_interactive_sliders_with_color_bar(title: str, valmin1, valmax1, valmin2, valmax2):
    fig, ((level_slider_axis, ax, color_bar_ax), (corner1, resolution_slider_axis, corner2)) \
        = plt.subplots(nrows=2,
                       ncols=3,
                       num=title,
                       figsize=(8, 5),
                       width_ratios=(2, 98, 3),
                       height_ratios=(98, 2),
                       tight_layout=True,
                       gridspec_kw={"left": 0.025, "right": 0.9, "bottom": 0.045, "top": 0.95, "hspace": 0.1,
                                    "wspace": 0.1})

    corner1.set_visible(False)
    corner2.set_visible(False)

    vertical_slider = Slider(ax=level_slider_axis, label="", orientation="vertical",
                             valmin=valmin1, valmax=valmax1, valinit=0, valstep=1,
                             color=(0.2, 0.2, 0.2))
    vertical_slider.valtext.set_visible(False)
    vertical_slider.label.set_visible(False)

    horizontal_slider = Slider(ax=resolution_slider_axis, label="",
                               valmin=valmin2, valmax=valmax2, valstep=1, valinit=16,
                               color=(0.2, 0.2, 0.2))
    horizontal_slider.valtext.set_visible(False)
    horizontal_slider.label.set_visible(False)

    return fig, ax, color_bar_ax, vertical_slider, horizontal_slider


def create_shared_2_column_plot(title: str):
    fig, (left_ax, right_ax) \
        = plt.subplots(nrows=1,
                       ncols=2,
                       num=title,
                       figsize=(8, 5),
                       sharey=True,
                       tight_layout=True)

    return fig, left_ax, right_ax


def create_2_column_plot(title: str):
    fig, (left_ax, right_ax) \
        = plt.subplots(nrows=1,
                       ncols=2,
                       num=title,
                       figsize=(8, 5),
                       tight_layout=True)

    return fig, left_ax, right_ax


def contourf(ax, levels, *args, **kwargs):
    return ax.contourf(*args, levels=levels, antialiased=False, algorithm="threaded", **kwargs)


def load_map(shape):
    coastlines = mpimg.imread("assets/equirectangular_projection.png")[::-1, :, 0]
    coastline_latitudes = np.linspace(0, shape[0], coastlines.shape[0])
    coastline_longitudes = np.linspace(0, shape[1], coastlines.shape[1])

    return coastlines, coastline_latitudes, coastline_longitudes


def draw_map(ax, coastlines, latitudes, longitudes):
    contourf(ax, 1, longitudes, latitudes, coastlines, colors=[(0, 0, 0, 0), (0, 0, 0, 0.2)])


def get_vmin_and_vmax(data):
    vmin = np.min(data)
    vmax = np.max(data)

    if vmin < 0 < vmax:
        if abs(vmin) > vmax:
            vmax = -vmin
        elif vmax > abs(vmin):
            vmin = -vmax

    return vmin, vmax


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


def plot_contour_at_time_and_level(filename: str, variable: str, time: int, level: int,
                                   folder: str = "compressed") -> None:
    data = load_variable_at_time_and_level(filename, variable, time, level, folder=folder)

    plt.contourf(np.linspace(-180, 180, 576), np.linspace(-90, 90, 361), data)

    plt.title(f"{variable} ({get_units_from_variable(variable)}) at {format_level(level)}"
              f" on {format_date(filename)} at {format_time(time)}", fontsize=8)
    plt.show()


def plot_interactive_variable_at_time(filename: str, variable: str, time: int, title: str, folder: str = "compressed",
                                      latitude_samples: int = 180, longitude_samples: int | None = None):
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
