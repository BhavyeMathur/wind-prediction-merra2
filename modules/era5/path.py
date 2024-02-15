import os

from modules.datetime import format_datetime


def era5_filepath(dt, output_folder: str = "ERA5/"):
    return f"{output_folder}/ERA5-tavg-{format_datetime(dt, pretty=False)}.nc"


def era5_file_exists(dt, output_folder: str = "ERA5/"):
    return os.path.isfile(era5_filepath(dt, output_folder))

