import numpy as np
import zlib


def encode(ia):
    ia = encode_difference_uint8(ia)
    ia = encode_zlib(ia, strategy=zlib.Z_FILTERED)
    return ia


def decode(ia):
    ia = decode_zlib(ia)
    ia = decode_difference_uint8(ia)
    return ia


def encode_zlib(ia, strategy: int = 1):
    compress = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS, zlib.DEF_MEM_LEVEL, strategy)
    deflated = compress.compress(ia)
    deflated += compress.flush()
    return np.frombuffer(deflated, dtype="uint8")


def decode_zlib(ia):
    decompress = zlib.decompressobj(-zlib.MAX_WBITS)
    inflated = decompress.decompress(ia)
    inflated += decompress.flush()
    return np.frombuffer(inflated, dtype="uint8")


def encode_run_length(ia):
    last_el = ia[0]
    out = [last_el]
    i = 0

    for el in ia:
        if (el == last_el) and i < 255:
            i += 1
        else:
            out.append(i)
            out.append(el)
            last_el = el
            i = 1
    out.append(i)
    assert sum(out[1::2]) == len(ia), f"{sum(out[1::2])} != {len(ia)}"
    return np.array(out, dtype="uint8")


def decode_run_length(ia):
    out = []
    for el, n in zip(ia[::2], ia[1::2]):
        out += [el] * n
    return np.array(out, dtype="uint8")


def encode_difference_uint8(ia):
    last_i = 0
    out = []

    # 255 - reset i to 0
    # 254 - increment i by 253

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
