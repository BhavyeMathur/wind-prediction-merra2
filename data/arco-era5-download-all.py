import os
import gc
from datetime import timedelta

from tqdm import tqdm

from modules.era5 import arco_era5 as era5
from modules.util import datetime_range, format_datetime


def download_at(month: int, day: int, hour: int, output_folder: str = "ERA5/wind"):
    wind_slice = era5.select_tavg_slice(ml_wind, 2012, 2022, (month, day, hour), verbose=False)
    wind_slice = era5.compute_tavg(wind_slice, verbose=False)
    wind_slice = era5.compress_dataset(wind_slice, verbose=False)

    era5.save_dataset(wind_slice, output_folder)


def download_all(start="01-01 00:00", end="12-31 23:00", output_folder: str = "ERA5/wind"):
    for dt in tqdm(datetime_range(start, end, timedelta(hours=1))):
        _, month, day, hour, *_ = dt.timetuple()

        if os.path.isfile(f"{output_folder}/ERA5-tavg-{format_datetime(month, day, hour)}.nc"):
            continue

        download_at(month, day, hour, output_folder=output_folder)
        gc.collect()


if __name__ == "__main__":
    ml_wind = era5.connect("gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
                           ("u_component_of_wind", "v_component_of_wind", "temperature", "vertical_velocity"))

    download_all(start="01-01 00:00", end="12-31 23:00")
