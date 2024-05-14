import xarray as xr

from modules.util import format_bytes


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
            coords[var] = xr.Variable(var, dataset[var].astype(AtmosphericVariable.get(var).dtype),
                                      attrs={"units": AtmosphericVariable.get(var).units})
            continue
        elif var == "time":
            coords[var] = dataset[var]

        compressed = dataset[var].values.astype("float16").view(view)
        variables[var] = xr.Variable(dataset.dims, compressed, attrs={"units": AtmosphericVariable.get(var).units})

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


def select_slice(dataset: xr.Dataset,
                 level: None | int = None, latitude: None | float = None, longitude: None | float = None) -> xr.Dataset:
    kwargs = {}
    if level is not None:
        kwargs["level"] = level
    if latitude is not None:
        kwargs["latitude"] = latitude
    if longitude is not None:
        kwargs["longitude"] = longitude

    return dataset.sel(**kwargs)


__all__ = ["compress_dataset", "uncompress_dataset", "select_slice"]

from .variables import AtmosphericVariable
