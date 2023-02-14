import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

import pandas as pd


means = {"U": 5.589, "V": 0.018}
stds = {"U": 9.832, "V": 3.232}

ESTIMATE_QUANTILE = 0.9935
TRAIN_TEST_SPLIT = 0.8
TRAIN_VAL_SPLIT = 0.9


class WindDataset(Dataset):
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

        self.y = self.x[["U", "V"]]

        del self.x["U"]
        del self.x["V"]

        self.x = self.x.values.astype("float16")
        self.y = self.y.values.astype("float16")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    @classmethod
    def init(cls):
        cls.data = pd.read_feather(f"../raw/subset/UV-NGCT-{ESTIMATE_QUANTILE}-{100000000}.ft")


def get_dense_model(input_size: int,
                    hidden_sizes: list[int],
                    output_size: int,
                    activation_func: callable):
    layers = []

    for size in hidden_sizes:
        layers.append(torch.nn.Linear(input_size, size))
        layers.append(activation_func())
        input_size = size

    layers.append(torch.nn.Linear(input_size, output_size))

    return torch.nn.Sequential(*layers)


class WindModel(pl.LightningModule):
    def __init__(self,
                 variable: str,
                 learning_rate: float,
                 loss_func: callable,
                 input_size: int,
                 hidden_sizes: list[int],
                 output_size: int,
                 activation_func: callable):

        super().__init__()

        self.learning_rate = learning_rate
        self.loss_func = loss_func

        self.variable = variable
        self.denorm_mult = stds[variable]
        self.denorm_add = means[variable]

        self.model = get_dense_model(input_size, hidden_sizes, output_size, activation_func)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        pred = self.model(x)
        loss = self.loss_func(pred, y)
        if self.loss_func == torch.nn.functional.mse_loss:
            mse = loss
        else:
            mse = torch.nn.functional.mse_loss(pred, y)

        self.log("train_loss", loss)
        self.log("train_rmse", (mse ** 0.5) * self.denorm_mult, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred = self.model(x)
        mse = torch.nn.functional.mse_loss(pred, y)
        mae = torch.nn.functional.l1_loss(pred, y)

        self.log("validation_rmse", (mse ** 0.5) * self.denorm_mult)
        self.log("validation_mae", mae * self.denorm_mult)

        return mse

    def test_step(self, batch, batch_idx):
        x, y = batch

        pred = self.model(x)
        mse = torch.nn.functional.mse_loss(pred, y)
        mae = torch.nn.functional.l1_loss(pred, y)

        self.log("test_rmse", (mse ** 0.5) * self.denorm_mult)
        self.log("test_mae", mae * self.denorm_mult)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, num_workers=1)

    @classmethod
    def init(cls, batch_size, train_data, test_data, validation_data):
        cls.batch_size = batch_size
        cls.train_data = train_data
        cls.test_data = test_data
        cls.validation_data = validation_data
