import os

from tqdm import tqdm

from modules.era5 import arco_era5 as era5
from modules.util import hours_in_year


def get_all_month_day_hour(start_month=1, start_day=1, start_hour=0,
                           end_month=12, end_day=31, end_hour=23):
    times = []

    for month, day, hour in hours_in_year():
        if month < start_month:
            continue
        if month == start_month and day < start_day:
            continue
        if month == start_month and day == start_day and hour < start_hour:
            continue

        times.append((month, day, hour))

        if month == end_month and day == end_day and hour == end_hour:
            break

    return times


def download_all(output_folder: str = "ERA5/wind", **kwargs):
    for month, day, hour in tqdm(get_all_month_day_hour(**kwargs)):
        if os.path.isfile(f"{output_folder}/ERA5-tavg-{month:02}{day:02}-{hour:02}00.nc"):
            continue

        wind_slice = era5.select_tavg_slice(ml_wind, 2012, 2022, f"{month:02}-{day:02} {hour:02}:00", verbose=False)
        wind_slice = era5.compute_tavg(wind_slice, verbose=False)
        wind_slice = era5.compress_dataset(wind_slice, verbose=False)

        era5.save_dataset(wind_slice, output_folder)
        del wind_slice


if __name__ == "__main__":
    ml_wind = era5.connect("gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
                           ("u_component_of_wind", "v_component_of_wind", "temperature", "vertical_velocity"))

    download_all(start_month=1, start_day=1, start_hour=0)
