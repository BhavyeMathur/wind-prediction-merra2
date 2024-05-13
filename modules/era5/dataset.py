import numpy as np
import xarray as xr

from modules.util import format_bytes


def get_latlon_contour_data(variable: str, datetime, level: int = None):
    if variable == "wind_speed" or variable == "wind_direction":
        u = get_latlon_contour_data("u_component_of_wind", datetime, level)
        v = get_latlon_contour_data("v_component_of_wind", datetime, level)
        if variable == "wind_speed":
            return np.sqrt(u ** 2 + v ** 2)
        else:
            return np.degrees(np.arctan2(u, v))

    if level is None:
        data = dataset[variable].values[:, ::-1]

    if dataset.attrs["is_float16"]:
        data = np.roll(data, 1440 // 2, axis=1)


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


def uncompress_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """
    Uncompresses the dataset from a netCDF-valid float16 format

    Args:
        dataset: the xarray dataset

    Returns:
        uncompressed xarray dataset
    """
    if not dataset.attrs.get("is_float16", False):
        return dataset  # input dataset isn't compressed as float16

    variables = {}

    var: str
    for var in dataset.variables:
        if var not in dataset.coords:
            uncompressed = dataset[var].values.view("float16").astype("float32")
            variables[var] = xr.Variable(dataset.dims, uncompressed, attrs=dataset[var].attrs)

    dataset = xr.Dataset(data_vars=variables, coords=dataset.coords, attrs=dataset.attrs | {"is_float16": 0})
    return dataset


__all__ = ["compress_dataset", "uncompress_dataset"]

from .variables import AtmosphericVariable
