import numpy as np
import pandas as pd
import scipy.interpolate as interp

from nc4 import *

data_cache = {}


def load_nc4_to_dataframe(filename: str, variable: str, folder: str = "compressed") -> pd.DataFrame:
    with open_xarray_dataset(filename, folder=folder) as dataset:
        packed_data = dataset.to_dataframe()[variable].array

    unpacked_data = packed_data.view("float16").astype("float16")
    dataframe = pd.DataFrame({variable: unpacked_data})

    with pd.option_context('mode.use_inf_as_na', True):
        dataframe.dropna(how="all", inplace=True)

    return dataframe


def load_variable_at_time(filename: str,
                          variable: str,
                          time: int,
                          folder: str = "compressed"):
    if (filename, variable, time) in data_cache:
        return data_cache[(filename, variable, time)]

    with open_xarray_dataset(filename, folder=folder) as dataset:
        data = np.array(dataset[variable][time])
        data = data.view("float16").astype("float16")

    data_cache[(filename, variable, time)] = data
    return data


def interpolate_variable_at_time(data: np.ndarray,
                                 latitude_samples: int,
                                 longitude_samples: int):
    interpolated_data = np.empty((data.shape[0], latitude_samples, longitude_samples), "float16")

    latitudes = np.linspace(0, data.shape[1], latitude_samples)
    longitudes = np.linspace(0, data.shape[2], longitude_samples)

    for lev in range(data.shape[0]):
        interpolation = interp.RectBivariateSpline(range(data.shape[1]), range(data.shape[2]), data[lev])
        interpolated_data[lev] = interpolation(latitudes, longitudes)

    return interpolated_data


def load_variable_at_time_and_level(filename: str,
                                    variable: str,
                                    time: int,
                                    level: int | float,
                                    folder: str = "compressed") -> np.array:
    if (filename, variable, time, level) in data_cache:
        return data_cache[(filename, variable, time, level)]

    if level % 1 != 0:
        data = load_variable_at_time_and_level(filename, variable, time, int(level), folder) * (1 - level % 1)
        data += load_variable_at_time_and_level(filename, variable, time, int(level) + 1, folder) * (level % 1)
        return data

    with open_xarray_dataset(filename, folder=folder) as dataset:
        data = np.array(dataset[variable, time, int(level)])

    data = data.view("float16").astype("float16")
    data_cache[(filename, variable, time, level)] = data
    return data


def interpolate_variable_at_time_and_level(data: np.ndarray,
                                           latitude_samples: int,
                                           longitude_samples: int):
    latitudes = np.linspace(0, data.shape[0], latitude_samples)
    longitudes = np.linspace(0, data.shape[1], longitude_samples)

    interpolation = interp.RectBivariateSpline(range(data.shape[0]), range(data.shape[1]), data)

    return interpolation(latitudes, longitudes)


def load_variable_at_time_and_latitude(filename: str,
                                       variable: str,
                                       time: int,
                                       latitude: int,
                                       folder: str = "compressed") -> np.array:
    with open_xarray_dataset(filename, folder=folder) as dataset:
        data = np.array(dataset[variable][time, :, latitude])

    data = data.view("float16").astype("float16")
    return data


def load_variable_at_time_level_and_latitude(filename: str,
                                             variable: str,
                                             time: int,
                                             level: int,
                                             latitude: int,
                                             folder: str = "compressed") -> np.array:
    with open_xarray_dataset(filename, folder=folder) as dataset:
        data = np.array(dataset[variable, time, level, latitude])

    data = data.view("float16").astype("float16")
    return data


def load_variable_at_time_level_and_longitude(filename: str,
                                              variable: str,
                                              time: int,
                                              level: int,
                                              longitude: int,
                                              folder: str = "compressed") -> np.array:
    with open_xarray_dataset(filename, folder=folder) as dataset:
        data = np.array(dataset[variable, time, level, :, longitude])

    data = data.view("float16").astype("float16")
    return data[:, 0]
