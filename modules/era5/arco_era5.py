import datetime

import xarray as xr
from dask.diagnostics import ProgressBar

from modules.util import format_bytes
from .metadata import *


def connect(url: str, variables: tuple[str, ...] = None, **kwargs) -> xr.Dataset:
    dataset = xr.open_zarr(url, **kwargs)

    if variables:
        dataset = dataset[list(variables)]

    print(f"Dataset size: {format_bytes(dataset.nbytes)}")
    return dataset


def select_tavg_slice(dataset: xr.Dataset, start_year: int, end_year: int, time: str,
                      start_level: int = 150, end_level: int = 1000) -> xr.Dataset:
    # dataset = dataset.convert_calendar("noleap")
    dataset = dataset.sel(level=slice(start_level, end_level),
                          time=slice(f"{start_year}-{time}", str(end_year), 24 * 365))

    dataset.attrs["tavg_start_year"] = start_year
    dataset.attrs["tavg_end_year"] = end_year
    dataset.attrs["datetime"] = time

    print(f"Dataset TAVG slice size: {format_bytes(dataset.nbytes)} ")
    return dataset


def compute_tavg(dataset: xr.Dataset) -> xr.Dataset:
    attrs = dataset.attrs

    with ProgressBar():
        dataset = dataset.mean("time").compute()

    dataset.attrs = attrs | {"is_tavg": 1}

    print(f"Dataset TAVG size: {format_bytes(dataset.nbytes)}")
    return dataset


def compress_dataset(dataset: xr.Dataset, view: str = "int16") -> xr.Dataset:
    variables = {}
    coords = {}

    for var in dataset.variables:
        if var in {"level", "latitude", "longitude"}:
            coords[var] = xr.Variable(var, dataset[var].astype(get_coord_dtype(var)),
                                      attrs={"units": get_units(var)})
            continue
        elif var == "time":
            coords[var] = dataset[var]

        compressed = dataset[var].values.astype("float16").view(view)
        variables[var] = xr.Variable(dataset.dims, compressed, attrs={"units": get_units(var)})

    ds = xr.Dataset(data_vars=variables, coords=coords, attrs=dataset.attrs | {"is_float16": 1})

    print(f"Compressed dataset size: {format_bytes(ds.nbytes)}")
    return ds


def save_dataset(dataset, output_folder: str) -> None:
    file = ["ERA5"]

    if dataset.attrs.get("is_tavg"):
        file.append("tavg")

        time = datetime.datetime.strptime(dataset.attrs["datetime"], "%m-%d %H:%M")
        file.append(f"{time.month:02}{time.day:02}")
        file.append(f"{time.hour:02}{time.minute:02}")
    else:
        raise NotImplementedError("Don't know how to save non-TAVG dataset")

    with ProgressBar():
        dataset.to_netcdf(f"{output_folder}/{'-'.join(file)}.nc")
