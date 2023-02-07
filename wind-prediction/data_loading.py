from calendar import monthrange

import numpy as np
import pandas as pd
import scipy.interpolate as interp
from tqdm.notebook import tqdm

from nc4 import *
from merra2 import *
from constants import COMPRESSED_FOLDER

data_cache = {}

YAVG_YEARS = (1980, 1981, 1982, 1983,
              1990, 1991, 1992, 1993,
              2000, 2001, 2002, 2003,
              2010, 2011, 2012, 2013)


def load_nc4_to_dataframe(filename: str, variable: str) -> pd.DataFrame:
    with open_xarray_dataset(filename, folder=COMPRESSED_FOLDER) as dataset:
        packed_data = dataset.to_dataframe()[variable].array

    unpacked_data = packed_data.view("float16").astype("float16")
    dataframe = pd.DataFrame({variable: unpacked_data})

    with pd.option_context('mode.use_inf_as_na', True):
        dataframe.dropna(how="all", inplace=True)

    return dataframe


def load_variable_on_day(filename: str,
                         variable: str):
    with open_xarray_dataset(filename, folder=COMPRESSED_FOLDER) as dataset:
        data = np.array(dataset[variable][:, -36:])
        data = data.view("float16").astype("float16")

    return data


def load_yavg_variable_on_day(filename: str,
                              *args,
                              years: tuple[int, ...] = YAVG_YEARS) -> np.array:
    data = np.zeros((8, 36, 361, 576), dtype="float64")
    for year in years:
        data += load_variable_on_day(filename.format(get_merra_stream_from_year(year), year), *args)

    return (data / len(years)).astype("float16")


def load_variable_at_time(filename: str,
                          variable: str,
                          time: int,
                          levels: int = 36):
    if (key := (filename, variable, time, None, None, None)) in data_cache:
        return data_cache[key][-levels:]

    with open_xarray_dataset(filename, folder=COMPRESSED_FOLDER) as dataset:
        data = np.array(dataset[variable][time, -levels:])
        data = data.view("float16").astype("float16")

    data_cache[key] = data
    return data


def load_yavg_variable_at_time(filename: str,
                               *args,
                               years: tuple[int, ...] = YAVG_YEARS,
                               levels: int = 36) -> np.array:
    data = np.zeros((levels, 361, 576), dtype="float64")
    for year in years:
        data += load_variable_at_time(filename.format(get_merra_stream_from_year(year), year), *args, levels=levels)

    return (data / len(years)).astype("float16")


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


def load_variable_at_level(filename: str,
                           variable: str,
                           level: int,
                           cache: bool = True):
    if (key := (filename, variable, None, level, None, None)) in data_cache:
        return data_cache[key]

    yyyy = get_year_from_filename(filename)
    data = np.zeros((365 * 8, 361, 576), dtype="float64")

    i = 0
    for mm in tqdm(range(1, 13)):
        for dd in range(1, monthrange(2001 if yyyy == "YAVG" else yyyy, mm)[1] + 1):
            if mm == 2 and dd == 29:
                continue  # let's just skip February 29th

            with open_xarray_dataset(filename.format(mm, dd), folder=COMPRESSED_FOLDER) as dataset:
                subdata = np.array(dataset[variable][:, level])
                data[i:i + 8] = subdata.view("float16").astype("float16")
                i += 8

    if cache:
        data_cache[key] = data
    return data


def load_yavg_variable_at_level(filename: str,
                                *args,
                                years: tuple[int, ...] = YAVG_YEARS):
    data = np.zeros((365 * 8, 361, 576), dtype="float64")

    for year in tqdm(years):
        path = filename.format(get_merra_stream_from_year(year), year)
        path = path.replace("mmdd", "{:0>2}{:0>2}")
        data += load_variable_at_level(path, *args)

    return (data / len(years)).astype("float16")


def load_variable_at_time_and_level(filename: str,
                                    variable: str,
                                    time: int,
                                    level: int | float,
                                    folder: str = COMPRESSED_FOLDER) -> np.array:
    if (key := (filename, variable, time, level, None, None)) in data_cache:
        return data_cache[key]

    if level % 1 != 0:
        data = load_variable_at_time_and_level(filename, variable, time, int(level)) * (1 - level % 1)
        data += load_variable_at_time_and_level(filename, variable, time, int(level) + 1) * (level % 1)
        return data

    if time % 1 != 0:
        data = load_variable_at_time_and_level(filename, variable, int(time), level) * (1 - time % 1)
        data += load_variable_at_time_and_level(filename, variable, int(time) + 1, level) * (time % 1)
        return data

    with open_xarray_dataset(filename, folder=folder) as dataset:
        if get_nc4_dimensions(filename, folder=folder) == 3:
            data = np.array(dataset[variable][int(time)])
        else:
            data = np.array(dataset[variable][int(time), int(level)])

    if is_nc4_packed_as_float32(filename, folder=folder):
        data = data.view("float16").astype("float16")

    data_cache[key] = data
    return data


def load_yavg_variable_at_time_and_level(filename: str,
                                         *args,
                                         years: tuple[int, ...] = YAVG_YEARS) -> np.array:
    data = np.zeros((361, 576), dtype="float64")
    for year in years:
        data += load_variable_at_time_and_level(filename.format(get_merra_stream_from_year(year), year), *args)

    return (data / len(years)).astype("float16")


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
                                       lat: int) -> np.array:
    if (key := (filename, variable, time, None, lat, None)) in data_cache:
        return data_cache[key]

    if lat % 1 != 0:
        data = load_variable_at_time_and_latitude(filename, variable, time, int(lat)) * (1 - lat % 1)
        data += load_variable_at_time_and_latitude(filename, variable, time, int(lat) + 1) * (lat % 1)
        return data

    with open_xarray_dataset(filename, folder=COMPRESSED_FOLDER) as dataset:
        data = np.array(dataset[variable][time, :, int(lat)])

    data = data.view("float16").astype("float16")
    data_cache[key] = data
    return data


def load_variable_at_time_and_longitude(filename: str,
                                        variable: str,
                                        time: int,
                                        lon: int) -> np.array:
    if (key := (filename, variable, time, None, None, lon)) in data_cache:
        return data_cache[key]

    if lon % 1 != 0:
        data = load_variable_at_time_and_longitude(filename, variable, time, int(lon)) * (1 - lon % 1)
        data += load_variable_at_time_and_longitude(filename, variable, time, int(lon) + 1) * (lon % 1)
        return data

    with open_xarray_dataset(filename, folder=COMPRESSED_FOLDER) as dataset:
        data = np.array(dataset[variable][time])

    data = data.view("float16").astype("float16")[:, :, int(lon)]
    data_cache[key] = data
    return data


def load_variable_at_time_level_and_latitude(filename: str,
                                             variable: str,
                                             time: int,
                                             level: int,
                                             latitude: int,
                                             folder=COMPRESSED_FOLDER) -> np.array:
    with open_xarray_dataset(filename, folder=folder) as dataset:
        data = np.array(dataset[variable][time, level, latitude])

    data = data.view("float16").astype("float16")
    return data


def load_yavg_variable_at_time_level_and_latitude(filename: str,
                                                  *args,
                                                  years: tuple[int, ...] = YAVG_YEARS) -> np.array:
    data = np.zeros((576,), dtype="float64")
    for year in years:
        data += load_variable_at_time_level_and_latitude(filename.format(get_merra_stream_from_year(year), year), *args)

    return (data / len(years)).astype("float16")


def load_variable_at_time_level_and_longitude(filename: str,
                                              variable: str,
                                              time: int,
                                              level: int,
                                              longitude: int) -> np.array:
    with open_xarray_dataset(filename, folder=COMPRESSED_FOLDER) as dataset:
        data = np.array(dataset[variable][time, level, :, longitude])

    data = data.view("float16").astype("float16")
    return data[:, 0]


def load_yavg_variable_at_time_level_and_longitude(filename: str,
                                                   *args,
                                                   years: tuple[int, ...] = YAVG_YEARS) -> np.array:
    data = np.zeros((361,), dtype="float64")
    for year in years:
        data += load_variable_at_time_level_and_longitude(filename.format(get_merra_stream_from_year(year), year),
                                                          *args)

    return (data / len(years)).astype("float16")
