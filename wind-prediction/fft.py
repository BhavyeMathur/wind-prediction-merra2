import numpy as np
import zlib


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


def encode_rlen(ia):
    out = []
    val_count = 0
    encoded_val = ia[0]

    for val in ia:
        if val == encoded_val:
            val_count += 1
        else:
            out.append(encoded_val)
            out.append(val_count)
            encoded_val = val
            val_count = 0

    if val_count != 0:
        out.append(encoded_val)
        out.append(val_count)

    return np.array(out, dtype="uint8")


def decode_rlen(ia):
    out = []
    for i in range(0, len(ia), 2):
        encoded_val = ia[i]
        for _ in range(ia[i + 1]):
            out.append(encoded_val)

    return np.array(out, dtype="uint16")


def encode_val_rlen(ia, encoded_val: int):
    out = []
    val_count = 0
    for val in ia:
        if val == encoded_val:
            val_count += 1
        else:
            out.append(val_count)
            out.append(val)
            val_count = 0

    if val_count != 0:
        out.append(val_count)

    return np.array(out, dtype="uint8")


def decode_val_rlen(ia, encoded_val: int):
    out = []
    for i in range(0, len(ia), 2):
        for _ in range(ia[i]):
            out.append(encoded_val)
        try:
            out.append(ia[i + 1])
        except IndexError:
            break

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


def idft3_at_level(fft_real, fft_imag, fft_i_indices, fft_j_indices, fft_k_indices):
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
