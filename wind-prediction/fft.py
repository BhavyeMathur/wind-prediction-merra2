import numpy as np
import numpy.fft._pocketfft_internal as pfi
import zlib


def irfft3_at_level(a: np.ndarray, lev: int):
    out = np.swapaxes(a, axis1=0, axis2=-1)
    out = pfi.execute(out, False, False, 1 / 36)  # a, is_real, is_forward, 1 / norm
    out = np.moveaxis(out, -1, 0)

    out = out[lev, :, :]
    out = pfi.execute(out, False, False, 1 / 361)  # a, is_real, is_forward, 1 / norm

    return np.fft.irfft(out.T, 576, axis=-1, norm=None)


def irfft3_at_level_and_latitude(a: np.ndarray, lev: int, lat: int):
    out = np.swapaxes(a, axis1=0, axis2=-1)
    out = pfi.execute(out, False, False, 1 / 36)  # a, is_real, is_forward, 1 / norm
    out = np.moveaxis(out, -1, 0)

    out = out[lev]
    out = pfi.execute(out, False, False, 1 / 361)  # a, is_real, is_forward, 1 / norm
    out = out.T[lat]

    return np.fft.irfft(out, 576, axis=-1, norm=None)


def encode_difference_uint8(ia):
    last_i = 0
    out = []

    for i in ia:
        while (di := i - last_i) > 253:
            out.append(254)
            last_i += 253

        if di < 0:
            out.append(255)
            last_i = 0

            while (di := i - last_i) > 253:
                out.append(254)
                last_i += 253

        last_i = i
        out.append(di)

    return np.array(out, dtype="uint8")


def decode_difference_uint8(ia):
    i = 0
    out = []

    for di in ia:
        if di == 254:
            i += 253
            continue

        if di == 255:
            i = 0
            continue

        i += di
        out.append(i)

    return np.array(out, dtype="uint16")


def encode_zlib(ia, strategy: int = 1):
    compress = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL, strategy)
    deflated = compress.compress(ia)
    deflated += compress.flush()
    return np.frombuffer(deflated, dtype="uint8")


def decode_zlib(ia, dtype="uint8"):
    decompress = zlib.decompressobj(-zlib.MAX_WBITS)
    inflated = decompress.decompress(ia)
    inflated += decompress.flush()
    return np.frombuffer(inflated, dtype=dtype)

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
        np.array(fft_indices, dtype="uint16")


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

    for j in range(289):
        for i in range(361):
            if amplitudes[i, j] < cutoff_amp:
                continue

            fft_real.append(fft[i, j].real)
            fft_imag.append(fft[i, j].imag)
            fft_i_indices.append(i)
            fft_j_indices.append(j)

    fft_real = (np.array(fft_real, dtype="float32") / 512).astype("float16")
    fft_real = encode_zlib(fft_real)

    fft_imag = (np.array(fft_imag, dtype="float32") / 512).astype("float16")
    fft_imag = encode_zlib(fft_imag)

    fft_i_indices = np.array(fft_i_indices, dtype="int16")
    fft_i_indices = encode_difference_uint8(fft_i_indices)
    fft_i_indices = encode_zlib(fft_i_indices)

    fft_j_indices = np.array(fft_j_indices, dtype="int16")
    fft_j_indices = encode_difference_uint8(fft_j_indices)
    fft_j_indices = encode_zlib(fft_j_indices, strategy=0)

    return fft_real, fft_imag, fft_i_indices, fft_j_indices


def idft2_at_time_and_level(fft_real, fft_imag, fft_i_indices, fft_j_indices):
    ifft = np.zeros((361, 289), dtype="complex64")

    fft_real = decode_zlib(fft_real, dtype="float16")
    fft_imag = decode_zlib(fft_imag, dtype="float16")
    fft = fft_real.astype("complex64") * 512 + fft_imag.astype("complex64") * 512j

    fft_i_indices = decode_zlib(fft_i_indices)
    fft_i_indices = decode_difference_uint8(fft_i_indices)

    fft_j_indices = decode_zlib(fft_j_indices)
    fft_j_indices = decode_difference_uint8(fft_j_indices)

    for idx in range(len(fft)):
        ifft[fft_i_indices[idx], fft_j_indices[idx]] = fft[idx]

    return np.fft.irfft2(ifft), len(fft)


def idft3_at_time(fft_real, fft_imag, fft_i_indices, fft_j_indices, fft_k_indices):
    ifft = np.zeros((36, 361, 289), dtype="complex64")

    fft_real = decode_zlib(fft_real, dtype="float16")
    fft_imag = decode_zlib(fft_imag, dtype="float16")
    fft = fft_real.astype("complex64") * 131072 + fft_imag.astype("complex64") * 131072j

    fft_i_indices = decode_zlib(fft_i_indices)
    fft_i_indices = decode_difference_uint8(fft_i_indices)

    fft_j_indices = decode_zlib(fft_j_indices)
    fft_j_indices = decode_difference_uint8(fft_j_indices)

    fft_k_indices = decode_zlib(fft_k_indices)
    fft_k_indices = decode_difference_uint8(fft_k_indices)

    for idx in range(len(fft)):
        ifft[fft_i_indices[idx], fft_j_indices[idx], fft_k_indices[idx]] = fft[idx]

    return np.fft.irfftn(ifft)


DFT3_LEVEL_CACHE = {}


def dft3_at_level(data: np.ndarray, level: int, quantile: float = 0.75, cache: bool = True):
    if level in DFT3_LEVEL_CACHE:
        fft, amplitudes = DFT3_LEVEL_CACHE[level]
    else:
        fft = np.fft.rfftn(data)
        amplitudes = np.abs(fft)

        if cache:
            DFT3_LEVEL_CACHE[level] = fft, amplitudes

    fft_real = []
    fft_imag = []
    fft_i_indices = []
    fft_j_indices = []
    fft_k_indices = []

    cutoff_amp = np.quantile(amplitudes, quantile)

    for k in range(289):
        for j in range(361):
            for i in range(365 * 8):
                if amplitudes[i, j, k] < cutoff_amp:
                    continue

                fft_real.append(fft[i, j, k].real)
                fft_imag.append(fft[i, j, k].imag)
                fft_i_indices.append(i)
                fft_j_indices.append(j)
                fft_k_indices.append(k)

    fft_real = (np.array(fft_real, dtype="float32") / 262144).astype("float16")
    fft_real = encode_zlib(fft_real)

    fft_imag = (np.array(fft_imag, dtype="float32") / 262144).astype("float16")
    fft_imag = encode_zlib(fft_imag)

    fft_i_indices = np.array(fft_i_indices, dtype="int16")
    fft_i_indices = encode_difference_uint8(fft_i_indices)
    fft_i_indices = encode_zlib(fft_i_indices)

    fft_j_indices = np.array(fft_j_indices, dtype="int16")
    fft_j_indices = encode_difference_uint8(fft_j_indices)
    fft_j_indices = encode_zlib(fft_j_indices)

    fft_k_indices = np.array(fft_k_indices, dtype="int16")
    fft_k_indices = encode_difference_uint8(fft_k_indices)
    fft_k_indices = encode_zlib(fft_k_indices)

    return fft_real, fft_imag, fft_i_indices, fft_j_indices, fft_k_indices


def idft3_at_level(fft_real, fft_imag, fft_i_indices, fft_j_indices, fft_k_indices):
    ifft = np.zeros((365 * 8, 361, 289), dtype="complex64")

    fft_real = decode_zlib(fft_real, dtype="float16")
    fft_imag = decode_zlib(fft_imag, dtype="float16")
    fft = fft_real.astype("complex64") * 262144 + fft_imag.astype("complex64") * 262144j

    fft_i_indices = decode_zlib(fft_i_indices)
    fft_i_indices = decode_difference_uint8(fft_i_indices)

    fft_j_indices = decode_zlib(fft_j_indices)
    fft_j_indices = decode_difference_uint8(fft_j_indices)

    fft_k_indices = decode_zlib(fft_k_indices)
    fft_k_indices = decode_difference_uint8(fft_k_indices)

    for idx in range(len(fft)):
        ifft[fft_i_indices[idx], fft_j_indices[idx], fft_k_indices[idx]] = fft[idx]

    return np.fft.irfftn(ifft), len(fft)
