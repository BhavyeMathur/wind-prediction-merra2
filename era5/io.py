import os
import xarray as xr

from dask.diagnostics import ProgressBar
from era5.util.datetime import datetime_func


# ERA5 = "/Volumes/Seagate Hub/ERA5/wind/"
ERA5 = "~/Downloads/"


def get_datalevel_from_datetime(datetime: str) -> str:
    if ":" in datetime:  # HH:MM
        return "hour"

    parts = datetime.split("-")
    if len(parts[0]) == 4:
        parts.pop(0)  # remove YYYY

    if len(parts) == 2:  # mm-DD
        return "day"
    if len(parts) == 1:  # mm
        return "month"
    else:
        return "year"


def datalevel_to_folder(datalevel: str) -> str:
    if datalevel == "hour":
        return "hourly"
    if datalevel == "day":
        return "daily"
    if datalevel == "month":
        return "monthly"
    if datalevel == "year":
        return ""


@datetime_func("datetime")
def era5_filename(datetime, is_tavg: bool = False, datalevel: str = "hour") -> str:
    if is_tavg or datetime.tavg:
        year = "tavg"
    else:
        year = datetime.year

    file = ["ERA5"]

    if datalevel == "hour":
        file.append(datetime.strftime(f"{year}-%m%d-%H%M"))
    elif datalevel == "day":
        file.append(datetime.strftime(f"{year}-%m%d"))
    elif datalevel == "month":
        file.append(datetime.strftime(f"{year}-%m"))

    return datalevel_to_folder(datalevel) + "/" + "-".join(file) + ".nc"


def save_dataset(dataset, output_folder: str = ERA5, verbose: bool = True) -> None:
    """
    Saves the ERA-5 dataset

    Args:
        dataset: the xarray dataset
        output_folder: filepath to output directory
        verbose: print debugging information?
    """

    filepath = f"""{output_folder}/{era5_filename(dataset.attrs['datetime'], dataset.attrs.get('is_tavg'), 
                                                datalevel=get_datalevel_from_datetime(dataset.attrs['datetime']))}"""
    if verbose:
        with ProgressBar():
            dataset.to_netcdf(filepath)
    else:
        dataset.to_netcdf(filepath)


datasets_cache: dict[str, xr.Dataset] = {}


@datetime_func("datetime")
def open_dataset(datetime, folder: str = ERA5, datalevel: str = "hour") -> xr.Dataset:
    """
    Opens an ERA-5 dataset

    Args:
        datetime: datetime corresponding to dataset file
        folder: filepath to ERA5 directory
        datalevel:
    """

    filepath = f"{folder}/{era5_filename(datetime, datalevel=datalevel)}"

    if filepath in datasets_cache:
        return datasets_cache[filepath]

    ds = xr.open_dataset(filepath)
    if datalevel == "hour":
        ds.coords["time"] = datetime
    else:
        try:
            ds = ds.sel(time=datetime)
        except TypeError:
            print(ds)
            raise
    datasets_cache[filepath] = ds
    return ds


def open_variable(variable: str | list[str], datetime, folder: str = ERA5, datalevel: str = "day") -> xr.Dataset:
    return open_dataset(datetime, folder, datalevel)[variable if isinstance(variable, list) else [variable]]


def era5_file_exists(time, folder: str = ERA5, datalevel: str = "hour"):
    """
    Checks if a locally saved ERA-5 file exists
    """
    return os.path.isfile(f"{folder}/{era5_filename(time, datalevel=datalevel)}")


__all__ = ["era5_filename", "save_dataset", "open_dataset", "open_variable", "era5_file_exists", "ERA5"]
