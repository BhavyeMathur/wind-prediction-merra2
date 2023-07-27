import torch
from torch.utils.data import Dataset

from maths import isa_temperature
from data_loading import *
from constants import MEANS, STDS


TRAIN_TEST_SPLIT = 0.8
TRAIN_VAL_SPLIT = 0.9


class WindDataset(Dataset):
    data: pd.DataFrame
    file: str

    def __init__(self, subset: str):
        if subset == "train":
            self.x = self.data.iloc[:int(len(self.data) * TRAIN_TEST_SPLIT)]
            self.x = self.data.iloc[:int(len(self.x) * TRAIN_VAL_SPLIT)]
        elif subset == "test":
            self.x = self.data.iloc[int(len(self.data) * TRAIN_TEST_SPLIT):]
        elif subset == "validation":
            self.x = self.data.iloc[:int(len(self.data) * TRAIN_TEST_SPLIT)]
            self.x = self.data.iloc[int(len(self.x) * TRAIN_VAL_SPLIT):]
        else:
            raise ValueError("Invalid Subset")

        if self.predict_difference and f"{self.variable}_est" in self.x.columns:
            self.y = self.x[self.variable] - self.x[self.variable + "_est"]
        else:
            self.y = self.x[self.variable]

        del self.x[self.variable]

        self.x = self.x.values.astype("float32")
        self.y = self.y.values.astype("float32")

        self.x = torch.tensor(self.x, device=self.device)
        self.y = torch.tensor(self.y, device=self.device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    @classmethod
    def init(cls, frac: float, quantile=0.9935, n=10000000, device="mps", variable="U",
             normalised=True, geographical=True, cyclic_time=True, cyclic_coordinates=False,
             combine_land=False, remove_lake=False, predict_difference=False, barometric=False, mathur2022=False):

        if quantile is None:
            df = pd.read_feather(f"subset/{variable}-{n}.ft")
        else:
            df = pd.read_feather(f"subset/{variable}-{quantile}-{n}.ft")

        indices = df[["lat", "lon"]].values.T

        cls.file = ""

        if barometric:
            cls.file += "B"

            df["isa_pressure"] = (100 * df["lev"].apply(get_pressure_from_level))
            df["altitude"] = df["isa_pressure"].apply(height_from_pressure)
            df["isa_temperature"] = df["altitude"].apply(isa_temperature)

            df["isa_pressure"] = ((df["isa_pressure"] - MEANS["P"]) / STDS["P"]).astype("float16")
            df["altitude"] = (df["altitude"] / 18000).astype("float16")
            df["isa_temperature"] = ((df["isa_temperature"] - MEANS["T"]) / STDS["T"]).astype("float16")

        if normalised:
            cls.file += "N"

            df["lat"] = (df["lat"] * 0.5) - 90
            df["lat"] /= 90
            df["lon"] = (df["lon"] * 0.625) - 180
            df["lon"] /= 180
            df["lev"] /= 36

            df["lev"] = df["lev"].astype("float16")
            df["lat"] = df["lat"].astype("float16")
            df["lon"] = df["lon"].astype("float16")

            df[variable] -= MEANS[variable]
            df[variable] /= STDS[variable]

            try:
                df[f"{variable}_est"] -= MEANS[variable]
                df[f"{variable}_est"] /= STDS[variable]
            except KeyError:
                pass

            df["day %"] = ((df["time"] % 8) / 8).astype("float16")
            df["year %"] = (df["time"] / 2920).astype("float16")
            del df["time"]

        if cyclic_time:
            cls.file += "CT"

            df["sin_day"] = np.sin(df["day %"] * np.pi * 2)
            df["cos_day"] = np.cos(df["day %"] * np.pi * 2)
            df["sin_year"] = np.sin(df["year %"] * np.pi * 2)
            df["cos_year"] = np.cos(df["year %"] * np.pi * 2)

        if cyclic_coordinates:
            cls.file += "CC"

            df["sin_lat"] = np.sin(df["lat"] * np.pi)
            df["cos_lat"] = np.cos(df["lat"] * np.pi)
            df["sin_lon"] = np.sin(df["lon"] * np.pi)
            df["cos_lon"] = np.cos(df["lon"] * np.pi)

        if geographical:
            cls.file += "G"

            frland = load_variable_at_time_and_level("MERRA2_101.const_2d_asm_Nx.00000000.nc4", variable="FRLAND",
                                                     time=0, level=0, folder="raw").astype("float16")
            frocean = load_variable_at_time_and_level("MERRA2_101.const_2d_asm_Nx.00000000.nc4", variable="FROCEAN",
                                                      time=0, level=0, folder="raw").astype("float16")
            frlake = load_variable_at_time_and_level("MERRA2_101.const_2d_asm_Nx.00000000.nc4", variable="FRLAKE",
                                                     time=0, level=0, folder="raw").astype("float16")
            frlandice = load_variable_at_time_and_level("MERRA2_101.const_2d_asm_Nx.00000000.nc4", variable="FRLANDICE",
                                                        time=0, level=0, folder="raw").astype("float16")
            phis = load_variable_at_time_and_level("MERRA2_101.const_2d_asm_Nx.00000000.nc4", variable="PHIS",
                                                   time=0, level=0, folder="raw")
            sgh = load_variable_at_time_and_level("MERRA2_101.const_2d_asm_Nx.00000000.nc4", variable="SGH",
                                                  time=0, level=0, folder="raw")

            phis -= phis.mean()
            phis /= phis.std()
            phis = phis.astype("float16")

            sgh -= sgh.mean()
            sgh /= sgh.std()
            sgh = sgh.astype("float16")

            if combine_land:
                cls.file += "CL"
                df["frland"] = frland[*indices] + frlandice[*indices]
            else:
                df["frland"] = frland[*indices]
                df["frlandice"] = frlandice[*indices]

            df["frocean"] = frocean[*indices]

            if not remove_lake:
                cls.file += "RL"
                df["frlake"] = frlake[*indices]

            df["phis"] = phis[*indices]
            df["sgh"] = sgh[*indices]

        if mathur2022:
            assert barometric
            assert normalised
            assert cyclic_time
            assert not cyclic_coordinates
            assert geographical
            assert not combine_land
            assert not remove_lake
            assert not predict_difference

            cls.file = "MATHUR2022"

        cls.device = device
        cls.variable = variable
        cls.predict_difference = predict_difference

        cls.data = df.sample(frac=frac)
