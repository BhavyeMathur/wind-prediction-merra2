"""Error Functions"""
import numpy as np


def me(data, prediction) -> float:
    return (data - prediction).mean()


def mae(data, prediction) -> float:
    return abs(data - prediction).mean()


def mape(data, prediction) -> float:
    return abs((data - prediction) / data).mean()


def wmape(data, prediction) -> float:
    return abs(data - prediction).sum() / abs(data).sum()


def smape(data, prediction) -> float:
    return np.nan_to_num(abs(prediction - data) / (abs(data) + abs(prediction)), False, nan=0).mean()


def mse(data, prediction) -> float:
    return ((data - prediction) ** 2).mean()


def rmse(data, prediction) -> float:
    return mse(data, prediction) ** 0.5


def r2(data, prediction):
    return 1 - ((data - prediction) ** 2).sum() / ((data - data.mean()) ** 2).sum()
