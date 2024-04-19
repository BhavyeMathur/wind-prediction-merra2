import xarray as xr
from tqdm import tqdm

from metplot import *
from modules.datetime import DateTime, dayofyear_to_date

from matplotlib.animation import FuncAnimation

ERA5 = "/Volumes/Seagate Hub/ERA5/wind"


def get_latlon_contour_data(variable: str, datetime, level):
    dataset = xr.open_dataset(f"{ERA5}/ERA5-{datetime}.nc")
    data = dataset[variable].sel(level=level).values

    if dataset.attrs["is_float16"]:
        data = data.view("float16")[::-1]
        data = np.roll(data, 1440 // 2, axis=1)

    if variable == "temperature":
        data -= 273.15

    return data


def plot_latlon_contour(variable: str, datetime, level, cmap=cmr.ocean, **kwargs):
    data = get_latlon_contour_data(variable, datetime, level)

    plot = ImagePlot(data, f"{format_variable(variable)} {format_time(datetime)} {format_pressure(level)}",
                     cmap=cmap, **kwargs)
    plot.plot()
    plot.show()


def animate_latlon_contour_vs_time(variable: str, level, cmap=cmr.ocean, **kwargs):
    datetime = DateTime(month=6, day=1, hour=14, year="tavg")
    d = get_latlon_contour_data(variable, datetime, level)

    plot = ImagePlot(d, f"{format_variable(variable)} {format_time(datetime)} {format_pressure(level)}",
                     cmap=cmap, **kwargs)
    plot.plot()

    def update(i):
        day, month = dayofyear_to_date(i // 24)
        dt = DateTime(month=month, day=day, hour=i % 24, year="tavg")

        data = get_latlon_contour_data(variable, dt, level)
        plot.title = f"{format_variable(variable)} {format_time(dt)} {format_pressure(level)}"
        return [plot.update_data(data)]

    anim = FuncAnimation(plot.fig, update, frames=tqdm(tuple(range(24 * 365))), blit=True, interval=0)
    anim.save(f"test.mp4", fps=30, writer="ffmpeg")


cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Temperature",
                                                           ["#d7e4fc", "#5fb1d4", "#4e9bc8", "#466ae1",
                                                            "#6b1966",
                                                            "#952c5e", "#d12d3e", "#fa7532", "#f5d25f"])


# plot_latlon_contour("temperature", DateTime(month=6, day=1, hour=14, year="tavg"), level=1000,
#                     cmap=cmap, diverging=True, vmin=-40, vmax=40)

animate_latlon_contour_vs_time("temperature", level=1000,
                               cmap=cmap, diverging=True, vmin=-40, vmax=40)
