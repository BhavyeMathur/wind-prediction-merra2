import numpy as np

import era5
from era5.util.util import format_bytes
from era5.maths.error import mae, rmse


class FourierRegression:
    def __init__(self, variable: era5.AtmosphericVariable, indices: list, quantile: float = 0.75):
        self._variable = variable
        self._indices = indices.copy()
        self._quantile = quantile

        self._dset = variable[*indices]
        self._data = self._dset.to_dataarray().values.squeeze()

        self._std = self._data.std()
        if self._std == 0:
            self._std = 1

        self._mean = self._data.mean()
        self._data = (self._data - self._mean) / self._std

        self._prediction = None
        self._fft_idxs = None
        self._fft_real = None
        self._fft_imag = None

    def fft(self):
        fft = np.fft.rfft(self._data)
        amplitudes = np.abs(fft)

        cutoff_amp = np.quantile(amplitudes, self._quantile)
        mask = amplitudes > cutoff_amp

        fft = fft[mask]

        self._fft_real = fft.real.astype("float16")
        self._fft_imag = fft.imag.astype("float16")
        self._fft_idxs = np.argwhere(mask).T.astype("uint16")

    def predict(self) -> np.ndarray:
        if self._prediction is None:
            fft = np.zeros(self._data.shape, dtype="complex64")
            fft[self._fft_idxs] = self._fft_real.astype("float32") + 1j * self._fft_imag.astype("float32")
            self._prediction = np.fft.irfft(fft, self._data.shape[0])

        return self._prediction * self._std + self._mean

    @property
    def nbytes(self) -> int:
        return self._fft_real.nbytes + self._fft_imag.nbytes + self._fft_idxs.nbytes

    def describe(self):
        prediction = self.predict()
        data = self._data * self._std + self._mean

        total_bytes = 2 * 24 * 365 * 25 * 721 * 1440
        input_bytes = self._data.nbytes // 2
        ratio = self.nbytes / input_bytes
        unit = self._variable.unit

        print(f"""
        Fourier Regression 1D:
            Data stdev: {data.std():.4f} {unit}
            MAE: {mae(data, prediction):.4f} {unit}
            RMSE: {rmse(data, prediction):.4f} {unit}
            
            Input size: {format_bytes(input_bytes)}
            Model size: {format_bytes(self.nbytes)}
            Size Ratio: {100 * ratio:.2f}%
            
            Frequencies: {len(self._fft_real.flatten())}
            Original size: {format_bytes(total_bytes)}
            Compressed size: {format_bytes(int(ratio * total_bytes))}
        """)
