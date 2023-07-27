import torch
from torch.utils.data import Dataset

import pandas as pd


means = {"U": 5.589, "V": 0.018}
stds = {"U": 9.832, "V": 3.232}

TRAIN_TEST_SPLIT = 0.8
TRAIN_VAL_SPLIT = 0.9

VARIABLE = "V"


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

        if self.predict_difference and f"{VARIABLE}_est" in self.x.columns:
            self.y = self.x[VARIABLE] - self.x[VARIABLE + "_est"]
        else:
            self.y = self.x[VARIABLE]

        for variable in self.variables:
            del self.x[variable]

        self.x = self.x.values.astype("float32")
        self.y = self.y.values.astype("float32")

        self.x = torch.tensor(self.x, device=self.device)
        self.y = torch.tensor(self.y, device=self.device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    @classmethod
    def init(cls, frac: float, quantile=0.9935, n=10000000, device="mps", variables=("U", "V"),
             normalised=True, geographical=True, cyclic_time=True, cyclic_coordinates=False,
             absolute=False, predict_difference=False, mathur2022=False, barometric=False, geographical_matrix=1):

        if mathur2022:
            file = "MATHUR2022"
            predict_difference = False
            absolute = False
        else:
            file = "N" if normalised else ""
            file += f"{geographical_matrix if geographical_matrix != 1 else ''}G" if geographical else ""
            file += "CT" if cyclic_time else ""
            file += "CC" if cyclic_coordinates else ""
            file += "B" if barometric else ""

        cls.device = device
        cls.predict_difference = predict_difference
        cls.variables = variables

        if quantile is None:
            file = f"../data/subset/{''.join(variables)}-{file}-{n}.ft"
        else:
            file = f"../data/subset/{''.join(variables)}-{file}-{quantile}-{n}.ft"
        cls.data = pd.read_feather(file)
        print("Loaded", file)

        cls.data = cls.data.sample(frac=frac)

        if absolute:
            for variable in variables:
                cls.data[variable] = cls.data[variable].abs()
                cls.data[f"{variable}_est"] = cls.data[f"{variable}_est"].abs()

            cls.data["V"] = cls.data["V"].abs()
            cls.data["V_est"] = cls.data["V_est"].abs()
