import os

import xarray as xr
import netCDF4 as nc

from util import *


def open_xarray_dataset(filename: str, folder: str = "raw") -> xr.Dataset:
    if not os.path.isdir(folder):
        os.makedirs(folder)

    filepath = f"{folder}/{filename}"

    log(f"Loading {filepath}")
    return xr.open_dataset(filepath)


def open_nc4_dataset(filename: str, folder: str = "raw", mode: str = "r") -> nc.Dataset:
    if not os.path.isdir(folder):
        os.makedirs(folder)

    filepath = f"{folder}/{filename}"

    log(f"Loading {filepath}")

    if mode == "w" and os.path.isfile(filename):
        os.remove(filename)

    return nc.Dataset(filepath, mode=mode)


def print_nc4_metadata(filename: str) -> None:
    with open_xarray_dataset(filename) as dataset:
        print(dataset)


def print_variable_metadata(filename: str, variable: str) -> None:
    with open_xarray_dataset(filename, folder=f"compressed/{variable}") as dataset:
        print(dataset)
