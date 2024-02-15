import os

from modules.datetime import format_datetime


def era5_filepath(time, output_folder: str = "ERA5/"):
    """
    Returns the filepath pointing to a locally saved ERA-5 file
    """
    return f"{output_folder}/ERA5-tavg-{format_datetime(time, pretty=False)}.nc"


def era5_file_exists(time, output_folder: str = "ERA5/"):
    """
    Checks if a locally saved ERA-5 file exists
    """
    return os.path.isfile(era5_filepath(time, output_folder))

