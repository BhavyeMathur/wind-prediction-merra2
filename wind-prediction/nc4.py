import os

import xarray as xr
import netCDF4 as nc

from constants import COMPRESSED_FOLDER


def open_xarray_dataset(filename: str, folder: str = COMPRESSED_FOLDER) -> xr.Dataset:
    if not os.path.isdir(folder):
        os.makedirs(folder)

    filepath = f"{folder}/{filename}"
    if os.path.exists(filepath):
        return xr.open_dataset(filepath)

    filepath = f"{folder}/{filename.split('.')[-2][:4]}/{filename}"
    return xr.open_dataset(filepath)


def open_nc4_dataset(filename: str, folder: str = "raw", mode: str = "r") -> nc.Dataset:
    if not os.path.isdir(folder):
        os.makedirs(folder)

    filepath = f"{folder}/{filename}"

    if mode == "w" and os.path.isfile(filename):
        os.remove(filename)

    return nc.Dataset(filepath, mode=mode)


def print_nc4_metadata(filename: str, folder: str = COMPRESSED_FOLDER) -> None:
    with open_xarray_dataset(filename, folder) as dataset:
        print(dataset)


def get_nc4_dimensions(filename: str, folder: str = COMPRESSED_FOLDER) -> int:
    with open_xarray_dataset(filename, folder) as dataset:
        return len(dataset.dims)


def get_nc4_dimension_size(filename: str, dimension: str, folder: str = COMPRESSED_FOLDER) -> int:
    with open_xarray_dataset(filename, folder) as dataset:
        return dataset.dims[dimension]


def is_nc4_packed_as_float32(filename: str, folder: str) -> bool:
    with open_xarray_dataset(filename, folder) as dataset:
        return dataset.dims["lon"] == 288


def print_file_metadata(filename: str, folder: str) -> None:
    with open_xarray_dataset(filename, folder=folder) as dataset:
        print(dataset)


def print_variable_metadata(filename: str, variable: str, folder: str) -> None:
    with open_xarray_dataset(filename, folder=folder) as dataset:
        print(dataset[variable])
