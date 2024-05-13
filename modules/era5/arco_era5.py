import xarray as xr
from dask.diagnostics import ProgressBar

from modules.util import format_bytes
from modules.datetime import format_datetime, datetime_func
from .io import save_dataset
from .dataset import compress_dataset


def connect(path: str, variables: tuple[str, ...] = None, verbose: bool = True, **kwargs) -> xr.Dataset:
    """
    Connects to a ZARR database and extracts certain variables

    Args:
        path: filepath or URL to ZARR file
        variables: tuple of variables to extract
        verbose: print debugging information?
        **kwargs: additional arguments to xarray.open_zarr() function
    """
    dataset = xr.open_zarr(path, **kwargs)

    if variables:
        dataset = dataset[list(variables)]

    if verbose:
        print(f"Dataset size: {format_bytes(dataset.nbytes)}")
    return dataset


@datetime_func("time")
def select_tavg_slice(dataset: xr.Dataset, start_year: int, end_year: int, time, verbose: bool = True) -> xr.Dataset:
    """
    Selects a slice of the dataset at a particular date & time, across several years.

    Args:
        dataset: the xarray dataset
        start_year: the year to begin the slice
        end_year: the year to end the slice
        time: 'mm-dd HH:MM' string or datetime object at which to select the data
        verbose: print debugging information?
    """
    time = format_datetime(time, pretty=True)

    dataset = dataset.sel(time=slice(f"{start_year}-{time}", str(end_year), 24 * 365))
    dataset.attrs |= {"tavg_start_year": start_year, "tavg_end_year": end_year, "datetime": time}

    if verbose:
        print(f"Dataset TAVG slice size: {format_bytes(dataset.nbytes)} ")
    return dataset


def select_vertical_slice(dataset: xr.Dataset, start_level: int = 150, end_level: int = 1000, verbose: bool = True):
    """
    Selects a vertical slice of the dataset

    Args:
        dataset: the xarray dataset
        start_level: in hPa
        end_level: in hPa
        verbose: print debugging information?
    """

    dataset = dataset.sel(level=slice(start_level, end_level))

    if verbose:
        print(f"Dataset TAVG slice size: {format_bytes(dataset.nbytes)} ")
    return dataset


def compute_tavg(dataset: xr.Dataset, verbose: bool = True) -> xr.Dataset:
    """
    Computes the time-average of a xarray dataset

    Args:
        dataset: the xarray dataset
        verbose: print debugging information?
    """
    attrs = dataset.attrs

    with ProgressBar():
        dataset = dataset.mean("time").compute()

    dataset.attrs = attrs | {"is_tavg": 1}

    if verbose:
        print(f"Dataset TAVG size: {format_bytes(dataset.nbytes)}")
    return dataset


__all__ = ["connect", "select_tavg_slice", "select_vertical_slice", "compute_tavg", "save_dataset", "compress_dataset"]
