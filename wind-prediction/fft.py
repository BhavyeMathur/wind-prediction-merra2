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
