import pandas as pd
import numpy as np

from nc4 import *


def save_df_as_parquet(df, output, compression_level=11):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df.to_parquet(output,
                  engine="fastparquet",
                  compression={"_default": {"type": "BROTLI", "args": {"level": compression_level}}})


def save_nc4_as_parquet(filename, variables, compression_level=11):
    output = filename[:-3] + "parquet"

    with open_xarray_dataset(filename) as dataset:
        df = dataset.to_dataframe()

    for var in _get_variables(variables):
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
