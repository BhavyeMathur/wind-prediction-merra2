import os

from dask.diagnostics import ProgressBar
from modules.datetime import parse_datetime, format_datetime


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


__all__ = ["era5_filename", "save_dataset", "era5_file_exists"]
