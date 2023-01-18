import os

import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc

from util import *


def open_xarray_dataset(filename, folder="raw") -> xr.Dataset:
    if not os.path.isdir(folder):
        os.makedirs(folder)

    filepath = f"{folder}/{filename}"

    log(f"Loading {filepath}")
    return xr.open_dataset(filepath)


def open_nc4_dataset(filename, folder="raw", mode="r") -> nc.Dataset:
    if not os.path.isdir(folder):
        os.makedirs(folder)

    filepath = f"{folder}/{filename}"

    log(f"Loading {filepath}")
    return nc.Dataset(filepath, mode=mode)


def print_nc4_metadata(filename):
    with open_xarray_dataset(filename) as dataset:
        print(dataset)


def save_df_as_parquet(df, output, compression_level=11):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df.to_parquet(output, engine="fastparquet",
                  compression={"_default": {"type": "BROTLI", "args": {"level": compression_level}}})


def save_nc4_as_parquet(filename, variables, compression_level=11):
    output = filename[:-3] + "parquet"

    with open_xarray_dataset(filename) as dataset:
        df = dataset.to_dataframe()

    for var in get_variables(variables):
        print(f"\tLoading '{var}'")
        variable = df[var]
        variable.reset_index(inplace=True, drop=True)

        variable16 = variable.astype("float16")
        print(f"\tStandard Deviation (SD): '{variable.std()}'")
        print(f"\tSD of float32 - float16: '{(variable - variable16).std()}'")

        print(f"\tSaving '{var}'\n")

        output_dir = f"dataframes/{var}"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        save_df_as_parquet(variable16, f"{output_dir}/{output}", compression_level)


def get_variables(variables) -> list[str]:
    if isinstance(variables, str):
        return [variables]
    return variables


def compress_nc4(filename, variables, compression_level=9):
    with open_nc4_dataset(filename) as dataset:
        for variable in get_variables(variables):
            with open_nc4_dataset(filename, folder=f"compressed/{variable}", mode="r") as dst:
                dst.createDimension("lon", 288)  # 588/2 because we pack float16s as float32s
                dst.createDimension("lat", 361)
                dst.createDimension("lev", 42)
                dst.createDimension("time", 8)

                data = dataset.variables[variable]

                dst.createVariable(variable, "f", data.dimensions, chunksizes=(1, 1, 361, 288), compression="zlib",
                                   complevel=compression_level)
                dst[variable].setncatts(data.__dict__)  # copy variable attributes via a dictionary

                data_float16 = np.array(data[:]).astype("float16")
                packed_float32 = data_float16.view("float32")
                dst[variable][:] = packed_float32
