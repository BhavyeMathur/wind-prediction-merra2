"""Interpolation Functions"""


import math


def linear_interpolate(values, index: int, t: float):
    return values[index] * (1 - t) + values[index + 1] * t


def cosine_interpolate(values, index: int, t: float):
    return linear_interpolate(values, index, (1 - math.cos(t * math.pi)) / 2)


def cubic_interpolate(values, index: int, t: float):
    p0 = values[max(0, index - 1)]
    p1 = values[index]
    p2 = values[index + 1]
    p3 = values[min(len(values) - 1, index + 2)]

    t2 = t * t
    a0 = p3 - p2 - p0 + p1
    a1 = p0 - p1 - a0
    a2 = p2 - p0
    a3 = p1

    return a0 * t * t2 + a1 * t2 + a2 * t + a3


def catmull_rom_interpolate(values, index: int, t: float):
    p0 = values[max(0, index - 1)]
    p1 = values[index]
    p2 = values[index + 1]
    p3 = values[min(len(values) - 1, index + 2)]

    t2 = t * t

    a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
    a1 = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
    a2 = -0.5 * p0 + 0.5 * p2
    a3 = p1

    return a0 * t * t2 + a1 * t2 + a2 * t + a3


def hermite_interpolate(values, index: int, t: float, tension: float = 0, bias: float = 0):
    p0 = values[max(0, index - 1)]
    p1 = values[index]
    p2 = values[index + 1]
    p3 = values[min(len(values) - 1, index + 2)]

    t2 = t * t
    t3 = t2 * t

    p2_minus_p1 = p2 - p1

    m0 = (p1 - p0) * (1 + bias)
    m0 += p2_minus_p1 * (1 - bias)

    m1 = p2_minus_p1 * (1 + bias)
    m1 += (p3 - p2) * (1 - bias)

    a0 = 2 * t3 - 3 * t2 + 1
    a1 = t3 - 2 * t2 + t
    a2 = t3 - t2
    a3 = -2 * t3 + 3 * t2

    return a0 * p1 + (1 - tension) * (a1 * m0 + a2 * m1) / 2 + a3 * p2


# TODO - Remove both of these functions
def get_hermite_interpolate(tension: float = 0, bias: float = 0):
    def _interpolate(*args):
        return hermite_interpolate(*args, tension=tension, bias=bias)

    return _interpolate


def fit_kochanek_bartels_spline(data, values, index: int, t: float, learning_rate: float = 0.5, iterations: int = 15):
    p0 = values[max(0, index - 1)]
    p1 = values[index]
    p2 = values[index + 1]
    p3 = values[min(len(values) - 1, index + 2)]

    t2 = t * t
    t3 = t2 * t

    best_loss = float("inf")
    best_tension = 0
    best_bias = 0

    a0 = 2 * t3 - 3 * t2 + 1
    a1 = (t3 - 2 * t2 + t) / 2
    a2 = (t3 - t2) / 2
    a3 = -2 * t3 + 3 * t2

    const = a0 * p1 + a3 * p2
    baseline = data - const

    m00 = p1 - p0
    m01 = p2 - p1
    m10 = m01
    m11 = p3 - p2

    bias = 0
    tension = 0

    for _ in range(iterations):
        m0 = m00 * (1 + bias) + m01 * (1 - bias)
        m1 = m10 * (1 + bias) + m11 * (1 - bias)

        if (loss := ((baseline - (a1 * m0 + a2 * m1) * (1 - tension)) ** 2).sum()) < best_loss:
            best_loss = loss
            best_tension = tension
        else:
            learning_rate /= 1.5

        d_outer = 2 * (baseline - (a1 * m0 + a2 * m1) * (1 - tension))
        dt = (d_outer * (a1 * m0 + a2 * m1)).mean()
        tension -= learning_rate * dt

    return best_tension, best_bias
