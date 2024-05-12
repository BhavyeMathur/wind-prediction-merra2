import numpy as np


def get_latlon_contour_data(variable: str, datetime, level: int = None):
    dataset = open_dataset(datetime)

    if variable == "wind_speed" or variable == "wind_direction":
        u = get_latlon_contour_data("u_component_of_wind", datetime, level)
        v = get_latlon_contour_data("v_component_of_wind", datetime, level)
        if variable == "wind_speed":
            return np.sqrt(u ** 2 + v ** 2)
        else:
            return np.degrees(np.arctan2(u, v))

    if level is None:
        data = dataset[variable].values[:, ::-1]
    else:
        data = dataset[variable].sel(level=level).values[::-1]

    if dataset.attrs["is_float16"]:
        data = data.view("float16")
        data = np.roll(data, 1440 // 2, axis=1)

    if variable == "temperature":
        data -= 273.15  # Kelvin to Celsius

    if variable == "vertical_velocity":
        data = np.sign(data) * (abs(data) ** 0.5)

    dataset.close()
    return data
