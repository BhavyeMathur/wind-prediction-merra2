import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

import pandas as pd


means = {"U": 5.589, "V": 0.018}
stds = {"U": 9.832, "V": 3.232}

ESTIMATE_QUANTILE = 0.9935
N = 100000000

TRAIN_TEST_SPLIT = 0.8
TRAIN_VAL_SPLIT = 0.9

DEVICE = "cpu"

NORMALIZED = True
GEOGRAPHICAL = True
CYCLIC_TIME = True
ABSOLUTE_U = True
ABSOLUTE_V = True

VARIABLE = "U"
DATASET = "N" if NORMALIZED else ""
DATASET += "G" if GEOGRAPHICAL else ""
DATASET += "CT" if CYCLIC_TIME else ""
DATASET += "AU" if ABSOLUTE_U else ""
DATASET += "AV" if ABSOLUTE_V else ""


class WindDataset(Dataset):
    data: pd.DataFrame

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

        self.y = self.x[VARIABLE]

        del self.x["U"]
        del self.x["V"]

        self.x = self.x.values.astype("float32")
        self.y = self.y.values.astype("float32")

        self.x = torch.Tensor(self.x).to(DEVICE)
        self.y = torch.Tensor(self.y).to(DEVICE)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    @classmethod
    def init(cls, frac: float):
        file = "N" if NORMALIZED else ""
        file += "G" if GEOGRAPHICAL else ""
        file += "CT" if CYCLIC_TIME else ""

        cls.data = pd.read_feather(f"../raw/subset/UV-{file}-{ESTIMATE_QUANTILE}-{100000000}.ft").sample(frac=frac)

        if ABSOLUTE_U:
            cls.data["U"] = cls.data["U"].abs()
            cls.data["U_est"] = cls.data["U_est"].abs()
        if ABSOLUTE_V:
            cls.data["V"] = cls.data["V"].abs()
            cls.data["V_est"] = cls.data["V_est"].abs()
