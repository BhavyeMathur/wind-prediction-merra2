import numpy as np

import era5
from era5.util.util import format_bytes
from era5.util.encoding import *
from era5.maths.error import mae, rmse


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
        prediction = self.predict()
        data = self._data

        total_bytes = 2 * 24 * 365 * 25 * 721 * 1440
        input_bytes = self._data.nbytes // 2
        ratio = self.nbytes / input_bytes
        unit = self._variable.unit

        print(f"""
        Fourier Regression {len(self._data.shape)}D:
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
