import torch
from torch.utils.data import Dataset

import pandas as pd


means = {"U": 5.589, "V": 0.018}
stds = {"U": 9.832, "V": 3.232}

TRAIN_TEST_SPLIT = 0.8
TRAIN_VAL_SPLIT = 0.9

VARIABLE = "U"


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

        if self.predict_difference:
            self.y = self.x[VARIABLE] - self.x[VARIABLE + "_est"]
        else:
            self.y = self.x[VARIABLE]

        del self.x["U"]
        del self.x["V"]

        self.x = self.x.values.astype("float32")
        self.y = self.y.values.astype("float32")

        self.x = torch.tensor(self.x, device=self.device)
        self.y = torch.tensor(self.y, device=self.device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    @classmethod
    def init(cls, frac: float, normalised=True, geographical=True, cyclic_time=True, quantile=0.9935, n=10000000,
             device="mps", absolute=False, predict_difference=False):

        file = "N" if normalised else ""
        file += "G" if geographical else ""
        file += "CT" if cyclic_time else ""

        cls.device = device
        cls.predict_difference = predict_difference

        cls.data = pd.read_feather(f"../data/subset/UV-{file}-{quantile}-{n}.ft").sample(frac=frac)

        if absolute:
            cls.data["U"] = cls.data["U"].abs()
            cls.data["U_est"] = cls.data["U_est"].abs()

            cls.data["V"] = cls.data["V"].abs()
            cls.data["V_est"] = cls.data["V_est"].abs()
