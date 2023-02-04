import numpy as np


def dft_at_time_level_and_latitude(data: np.ndarray, quantile: float = 0.75):
    fft = np.fft.rfft(data)
    amplitudes = np.abs(fft)

    cutoff_amp = np.quantile(amplitudes, quantile)
    fft_real = []
    fft_imag = []
    fft_indices = []

    for i, val in enumerate(fft):
        if amplitudes[i] < cutoff_amp:
            continue

        fft_real.append(fft[i].real)
        fft_imag.append(fft[i].imag)
        fft_indices.append(i)

    return np.array(fft_real, dtype="float16"), np.array(fft_imag, dtype="float16"), \
        np.array(fft_indices, dtype="uint8")


def idft_at_time_level_and_latitude(fft_real, fft_imag, fft_indices):
    fft = np.zeros((289,), dtype="complex64")

    for k in range(len(fft_indices)):
        fft[fft_indices[k]] = fft_real[k] + 1j * fft_imag[k]

    return np.fft.irfft(fft)


def dft2_at_time_and_level(data: np.ndarray, quantile: float = 0.75):
    fft = np.fft.rfft2(data)
    amplitudes = np.abs(fft)

    fft_real = []
    fft_imag = []
    fft_i_indices = []
    fft_j_indices = []

    cutoff_amp = np.quantile(amplitudes, quantile)

    for i in range(361):
        for j in range(256):
            if amplitudes[i, j] < cutoff_amp:
                continue

            fft_real.append(fft[i, j].real / 512)
            fft_imag.append(fft[i, j].imag / 512)
            fft_i_indices.append(i)
            fft_j_indices.append(j)

    return np.array(fft_real, dtype="float16"), np.array(fft_imag, dtype="float16"), \
        np.array(fft_i_indices, dtype="uint16"), np.array(fft_j_indices, dtype="uint8")


def idft2_at_time_and_level(fft_real, fft_imag, fft_i_indices, fft_j_indices):
    fft = np.zeros((361, 289), dtype="complex64")

    for k in range(len(fft_i_indices)):
        fft[fft_i_indices[k], fft_j_indices[k]] = 512 * fft_real[k] + 512j * fft_imag[k]

    return np.fft.irfft2(fft)


def dft3_at_time(data: np.ndarray, quantile: float = 0.75):
    zf = np.fft.ifftshift(data)
    zf = np.fft.fftn(zf)
    zf = np.fft.fftshift(zf)
    amplitudes = np.abs(zf)

    zf_real = (zf.real / 512).astype("float16")
    zf_imag = (zf.imag / 512).astype("float16")

    cutoff_amp = np.quantile(amplitudes, quantile)
    transform = {(i, j, k): (zf_real[i, j, k], zf_imag[i, j, k]) for k in range(576) for j in range(361)
                 for i in range(72) if amplitudes[i, j, k] > cutoff_amp}

    return transform


DFT3_LEVEL_CACHE = {}


def dft3_at_level(data: np.ndarray, level: int, quantile: float = 0.75):
    if level in DFT3_LEVEL_CACHE:
        fft = DFT3_LEVEL_CACHE[level]
    else:
        fft = np.fft.rfftn(data)
        DFT3_LEVEL_CACHE[level] = fft

    amplitudes = np.abs(fft)

    fft_real = []
    fft_imag = []
    fft_i_indices = []
    fft_j_indices = []
    fft_k_indices = []

    cutoff_amp = np.quantile(amplitudes, quantile)

    last_i = 0
    last_j = 0
    last_k = 0

    for k in range(289):
        for j in range(361):
            for i in range(365 * 8):
                if amplitudes[i, j, k] < cutoff_amp:
                    continue

                fft_real.append(fft[i, j, k].real)
                fft_imag.append(fft[i, j, k].imag)

                while (di := i - last_i) > 253:
                    fft_i_indices.append(254)
                    last_i += 253

                while (dj := j - last_j) > 253:
                    fft_j_indices.append(254)
                    last_j += 253

                while (dk := k - last_k) > 253:
                    fft_k_indices.append(254)
                    last_k += 253

                if di < 0:
                    fft_i_indices.append(255)
                    last_i = 0
                else:
                    fft_i_indices.append(di)
                    last_i += di

                if dj < 0:
                    fft_j_indices.append(255)
                    last_j = 0
                else:
                    fft_j_indices.append(dj)
                    last_j += dj

                if dk < 0:
                    fft_k_indices.append(255)
                    last_k = 0
                else:
                    fft_k_indices.append(dk)
                    last_k += dk

    fft_real = np.array(fft_real, dtype="float32") / 32768
    fft_imag = np.array(fft_imag, dtype="float32") / 32768

    fft_i_indices = np.array(fft_i_indices, dtype="uint8")
    fft_j_indices = np.array(fft_j_indices, dtype="uint8")
    fft_k_indices = np.array(fft_k_indices, dtype="uint8")

    return fft_real.astype("float16"), fft_imag.astype("float16"), \
        fft_i_indices, fft_j_indices, fft_k_indices


def idft3_at_time_and_level(fft_real, fft_imag, fft_i_indices, fft_j_indices, fft_k_indices):
    ifft = np.zeros((365 * 8, 361, 289), dtype="complex64")
    fft = fft_real.astype("complex64") * 32768 + fft_imag.astype("complex64") * 32768j

    i = 0
    j = 0
    k = 0

    i_index = 0
    j_index = 0
    k_index = 0

    for idx in range(len(fft)):
        while (di := fft_i_indices[i_index]) == 254:
            i += 253
            i_index += 1

        while (dj := fft_j_indices[j_index]) == 254:
            j += 253
            j_index += 1

        while (dk := fft_k_indices[k_index]) == 254:
            k += 253
            k_index += 1

        if di == 255:
            i = 0
        else:
            i += di

        if dj == 255:
            j = 0
        else:
            j += dj

        if dk == 255:
            k = 0
        else:
            k += dk

        ifft[i, j, k] = fft[idx]

        i_index += 1
        j_index += 1
        k_index += 1

    return np.fft.irfftn(ifft)
