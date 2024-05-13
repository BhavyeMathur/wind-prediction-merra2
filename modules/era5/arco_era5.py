import os
import xarray as xr
from dask.diagnostics import ProgressBar

from modules.util import format_bytes
from modules.datetime import format_datetime, parse_datetime, datetime_func
from .variables import AtmosphericVariable


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


def compress_dataset(dataset: xr.Dataset, view: str = "int16", verbose: bool = True) -> xr.Dataset:
    """
    Compresses the dataset into a netCDF-valid float16 format

    Args:
        dataset: the xarray dataset
        view: compression dtype to use
        verbose: print debugging information?

    Returns:
        compressed xarray dataset
    """
    variables = {}
    coords = {}

    var: str
    for var in dataset.variables:
        if var in {"level", "latitude", "longitude"}:
            coords[var] = xr.Variable(var, dataset[var].astype(AtmosphericVariable.get_dtype(var)),
                                      attrs={"units": AtmosphericVariable.get_units(var)})
            continue
        elif var == "time":
            coords[var] = dataset[var]

        compressed = dataset[var].values.astype("float16").view(view)
        variables[var] = xr.Variable(dataset.dims, compressed, attrs={"units": AtmosphericVariable.get_units(var)})

    dataset = xr.Dataset(data_vars=variables, coords=coords, attrs=dataset.attrs | {"is_float16": 1})

    if verbose:
        print(f"Compressed dataset size: {format_bytes(dataset.nbytes)}")
    return dataset


def uncompress_dataset(dataset: xr.Dataset, verbose: bool = True) -> xr.Dataset:
    """
    Uncompresses the dataset from a netCDF-valid float16 format

    Args:
        dataset: the xarray dataset
        verbose: print debugging information?

    Returns:
        uncompressed xarray dataset
    """
    if not dataset.attrs.get("is_float16", False):
        return dataset  # input dataset isn't compressed as float16

    variables = {}
    coords = {}

    var: str
    for var in dataset.variables:
        if var in {"level", "latitude", "longitude"}:
            coords[var] = xr.Variable(var, dataset[var].astype(AtmosphericVariable.get_dtype(var)),
                                      attrs={"units": AtmosphericVariable.get_units(var)})
            continue

        elif var == "time":
            coords[var] = dataset[var]

        compressed = dataset[var].values.view("float16")
        variables[var] = xr.Variable(dataset.dims, compressed, attrs={"units": AtmosphericVariable.get_units(var)})

    dataset = xr.Dataset(data_vars=variables, coords=coords, attrs=dataset.attrs | {"is_float16": 0})

    if verbose:
        print(f"Compressed dataset size: {format_bytes(dataset.nbytes)}")
    return dataset


def era5_filename(datetime, is_tavg: bool = True) -> str:
    file = ["ERA5"]
    datetime = parse_datetime(datetime)

    if is_tavg:
        file.append("tavg")
        file.append(format_datetime(datetime, pretty=False))
    else:
        raise NotImplementedError("Non-TAVG datasets unimplemented")

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


def era5_file_exists(time, output_folder: str = "ERA5/"):
    """
    Checks if a locally saved ERA-5 file exists
    """
    return os.path.isfile(f"{output_folder}/{era5_filename(time)}")


__all__ = [connect, select_tavg_slice, select_vertical_slice, compute_tavg, compress_dataset, uncompress_dataset,
           era5_filename, save_dataset, era5_file_exists]
