"""Error Functions"""


def me(data, prediction):
    return (data - prediction).mean()


def mae(data, prediction):
    return abs(data - prediction).mean()


def mape(data, prediction):
    ((data - prediction) / data).mean()


def mse(data, prediction):
    return ((data - prediction) ** 2).mean()


def rmse(data, prediction):
    return mse(data, prediction) ** 0.5
