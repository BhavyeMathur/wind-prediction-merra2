import era5
from era5.util.util import format_bytes
from era5.variables.text import format_unit
from era5.util.encoding import *
from era5.maths.error import *


class FourierRegression:
    def __init__(self, variable: era5.AtmosphericVariable, indices: list, quantile: float = 0.75):
        self._variable = variable
        self._indices = indices.copy()
        self._quantile = quantile

        self._dset = variable[*indices]
        self._data = self._dset.to_dataarray().values.squeeze()

        self._prediction = None
        self._fft_idxs = None
        self._fft_real = None
        self._fft_imag = None

    def fft(self):
        fft = np.fft.rfftn(self._data, norm="forward")

        amplitudes = np.abs(fft)

        cutoff_amp = np.quantile(amplitudes, self._quantile)
        mask = amplitudes > cutoff_amp

        fft = fft[mask]

        self._fft_real = encode_zlib(fft.real.astype("float16"))
        self._fft_imag = encode_zlib(fft.imag.astype("float16"))
        self._fft_idxs = np.argwhere(mask).T
        self._fft_idxs = tuple(map(encode, self._fft_idxs))

    def predict(self) -> np.ndarray:
        if self._prediction is None:
            fft_idxs = np.array(list(map(decode, self._fft_idxs)))
            fft_real = decode_zlib(self._fft_real).view("float16")
            fft_imag = decode_zlib(self._fft_imag).view("float16")

            fft = np.zeros(self._data.shape, dtype="complex64")
            fft[*fft_idxs] = fft_real + 1j * fft_imag.astype("float32")
            self._prediction = np.fft.irfftn(fft, self._data.shape, norm="forward")

        return self._prediction

    def data(self) -> np.ndarray:
        return self._data

    @property
    def nbytes(self) -> int:
        return self._fft_real.nbytes + self._fft_imag.nbytes + sum(ar.nbytes for ar in self._fft_idxs)

    def describe(self):
        total_bytes = 2 * 24 * 365 * 25 * 721 * 1440
        ratio = self.nbytes / self.input_bytes

        print(f"""
        Fourier Regression {len(self._data.shape)}D:
            Data stdev: {self.std():.4f}{format_unit(self._variable.unit)}
            MAE: {self.mae():.4f}{format_unit(self._variable.unit)}
            RMSE: {self.rmse():.4f}{format_unit(self._variable.unit)}
            
            Input size: {format_bytes(self.input_bytes)}
            Model size: {format_bytes(self.nbytes)}
            Size Ratio: {100 * ratio:.2f}%
            
            Frequencies: {len(self._fft_real.flatten())}
            Original size: {format_bytes(total_bytes)}
            Compressed size: {format_bytes(int(ratio * total_bytes))}
        """)

    def std(self) -> float:
        return self._data.std()

    def var(self) -> float:
        return self._data.var()

    def mae(self) -> float:
        prediction = self.predict()
        return mae(self._data, prediction)

    def rmse(self) -> float:
        prediction = self.predict()
        return rmse(self._data, prediction)

    def mse(self) -> float:
        prediction = self.predict()
        return mse(self._data, prediction)

    def r2(self) -> float:
        prediction = self.predict()
        return r2(self._data, prediction)

    @property
    def input_bytes(self) -> int:
        return len(self._data.ravel()) * 2

    @property
    def variable(self) -> era5.AtmosphericVariable:
        return self._variable


def evaluate_ft(models: list[FourierRegression]) -> np.ndim:
    total_bytes = 2 * 24 * 365 * 25 * 721 * 1440
    model_size = 0
    input_size = 0

    predictions = []
    data = []

    for model in models:
        model.fft()
        predictions.append(model.predict())
        data.append(model.data())

        model_size += model.nbytes
        input_size += model.input_bytes

    predictions = np.array(predictions)
    data = np.array(data)

    variable = models[0].variable
    unit = format_unit(variable.unit)

    print(f"""
    Data stdev: {data.std():.4f}{unit}
    Data range: {data.min():.3f} to {data.max():.3f}{unit} ({data.max() - data.min():.4f}{unit})
    
    R: {(r := r2(data, predictions)) ** 0.5:.4f}
    R²: {r:.4f}
    MAE: {mae(data, predictions):.4f}{unit}
    RMSE: {rmse(data, predictions):.4f}{unit}
    
    MAPE: {100 * mape(data, predictions):.3f}%
    wMAPE: {100 * wmape(data, predictions):.3f}%
    SMAPE: {100 * smape(data, predictions):.3f}%
    
    Original size: {format_bytes(total_bytes)}
    Compressed size: {format_bytes(int(model_size / input_size * total_bytes))}
    """)

    return np.array(predictions)
