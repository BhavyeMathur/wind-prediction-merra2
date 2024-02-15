import gc
from datetime import timedelta

from tqdm import tqdm

import modules.era5 as era5
from modules.datetime import datetime_range, datetime_func


@datetime_func("dt")
def download_at(dt, output_folder: str = "ERA5/", start_year=2012, end_year=2022, verbose=True):
    wind_slice = era5.select_tavg_slice(ml_wind, start_year, end_year, dt, verbose=verbose)
    wind_slice = era5.compute_tavg(wind_slice, verbose=verbose)
    wind_slice = era5.compress_dataset(wind_slice, verbose=verbose)

    era5.save_dataset(wind_slice, output_folder)


@datetime_func("start", "end")
def download_all(start="01-01 00:00", end="12-31 23:00", output_folder: str = "ERA5/"):
    for dt in tqdm(datetime_range(start, end, timedelta(hours=1))):
        if era5.era5_file_exists(dt, output_folder):
            continue

        download_at(dt, output_folder, verbose=False)
        gc.collect()


if __name__ == "__main__":
    ml_wind = era5.connect("gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
                           ("u_component_of_wind", "v_component_of_wind", "temperature", "vertical_velocity"))

    download_all(start="01-01 00:00", end="12-31 23:00", output_folder="ERA5/wind")
