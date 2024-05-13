import os
import xarray as xr

from dask.diagnostics import ProgressBar
from modules.datetime import format_datetime, datetime_func


# ERA5 = "/Volumes/Seagate Hub/ERA5/wind"
ERA5 = "~/Downloads"


@datetime_func("datetime")
def era5_filename(datetime, is_tavg: bool = False) -> str:
    if is_tavg and not datetime.tavg:
        datetime.tavg = True

    file = ["ERA5", format_datetime(datetime, pretty=False)]
    return "-".join(file) + ".nc"


def save_dataset(dataset, output_folder: str, verbose: bool = True) -> None:
    """
    Saves the ERA-5 dataset

    Args:
        dataset: the xarray dataset
        output_folder: filepath to output directory
        verbose: print debugging information?

    Raises:
        NotImplementedError: dataset must be time-averaged
    """

    filepath = f"{output_folder}/{era5_filename(dataset.attrs['datetime'], dataset.attrs.get('is_tavg'))}"
    if verbose:
        with ProgressBar():
            dataset.to_netcdf(filepath)
    else:
        dataset.to_netcdf(filepath)


@datetime_func("datetime")
def open_dataset(datetime, folder: str = ERA5) -> xr.Dataset:
    """
    Opens an ERA-5 dataset

    Args:
        datetime: datetime corresponding to dataset file
        folder: filepath to ERA5 directory

    Raises:
        NotImplementedError: dataset must be time-averaged
    """

    filepath = f"{folder}/{era5_filename(datetime)}"
    return xr.open_dataset(filepath)


def era5_file_exists(time, output_folder: str = "ERA5/"):
    """
    Checks if a locally saved ERA-5 file exists
    """
    return os.path.isfile(f"{output_folder}/{era5_filename(time)}")


__all__ = ["era5_filename", "save_dataset", "open_dataset", "era5_file_exists"]