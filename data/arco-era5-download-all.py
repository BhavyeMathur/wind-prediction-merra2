import os

from tqdm import tqdm

from modules.era5 import arco_era5 as era5
from modules.util import hours_in_year

ml_wind = era5.connect("gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
                       ("u_component_of_wind", "v_component_of_wind", "temperature", "vertical_velocity"))

for month, day, hour in tqdm(hours_in_year(), total=365 * 24):
    if os.path.isfile(f"ERA5/wind/ERA5-tavg-{month:02}{day:02}-{hour:02}00.nc"):
        continue

    wind_slice = era5.select_tavg_slice(ml_wind, 2012, 2022, f"{month:02}-{day:02} {hour:02}:00")
    wind_slice = era5.compute_tavg(wind_slice)
    wind_slice = era5.compress_dataset(wind_slice)
    era5.save_dataset(wind_slice, "ERA5/wind")
