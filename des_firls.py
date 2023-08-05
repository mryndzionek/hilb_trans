import math

import numpy as np

from scipy import linalg
import scipy.signal as sig
from scipy.fftpack import fft, fftshift

import matplotlib.pyplot as plt

from functools import partial


def plot(taps, fs, des, axs, method, color='blue', alpha=1.0):

    axs[0].step(range(len(taps)), taps,
                where='mid', color=color, marker='.', alpha=alpha, label=method)
    axs[0].set_xlim(-1, len(taps))
    # axs[0].set_xlabel('n')
    axs[0].set_ylabel('hn_b')

    A = fft(taps, 2048)
    freq = np.linspace(-0.5, 0.5, len(A), endpoint=False)
    A = fftshift(A)
    f_response = 20 * np.log10(np.abs(A))
    p_response = np.unwrap(np.angle(A))

    if not fs is None:
        axs[1].plot(fs, des, marker='x', markersize=5.0,
                    color='red', alpha=1.0, linestyle="None")
    axs[1].plot(freq, np.abs(A), color=color, alpha=alpha, label=method)
    axs[1].set_xlim(-0.5, 0.5)
    axs[1].set_xticks(np.arange(-0.5, 0.5, 0.02), minor=True)
    axs[1].set_yticks(np.arange(-1.0, 2.0, 0.4), minor=True)
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Gain')

    if not fs is None:
        axs[2].plot(fs, 20 * np.log10(des), marker='x', markersize=5.0,
                    color='red', alpha=1.0, linestyle="None")

    axs[2].plot(freq, f_response, color=color, alpha=alpha, label=method)
    axs[2].set_xlim(-0.5, 0.5)
    axs[2].set_xticks(np.arange(-0.5, 0.5, 0.02), minor=True)
    yt_min = min(f_response[f_response > -420]) - 10
    axs[2].set_ylim(yt_min, 20.0)
    axs[2].set_yticks(np.arange(yt_min, 10.0, (10 - yt_min) / 5))
    axs[2].set_xlabel('Frequency')
    axs[2].set_ylabel('Gain (dB)')

    axs[3].plot(freq, p_response, color=color, alpha=alpha, label=method)
    axs[3].set_xlim(-0.5, 0.5)
    axs[3].set_xticks(np.arange(-0.5, 0.5, 0.02), minor=True)
    axs[3].set_yticks(np.arange(min(p_response), max(
        p_response), (max(p_response) - min(p_response)) / 5))
    axs[3].set_xlabel('Frequency')
    axs[3].set_ylabel('Phase (radians)')


def linear_ramp(xs, ys, x):
    assert(len(xs) == len(ys))

    for i, x_curr in enumerate(xs):
        if x < x_curr:
            break

    x_prev = xs[i - 1] if i > 0 else 0.0
    y_curr = ys[i]
    y_prev = ys[i - 1] if i > 0 else 0.0

    slope = (y_curr - y_prev) / (x_curr - x_prev)
    return y_prev + ((x - x_prev) * slope)


def sin_ramp(xs, ys, x):
    assert(len(xs) == len(ys))

    for i, x_curr in enumerate(xs):
        if x < x_curr:
            break

    x_prev = xs[i - 1] if i > 0 else 0.0
    y_curr = ys[i]
    y_prev = ys[i - 1] if i > 0 else 0.0

    if not math.isclose(y_curr, y_prev):
        def slope_f(x):
            return math.sin(x)

        w = x_curr - x_prev
        h = abs(y_curr - y_prev)
        x_t = (math.pi / w) * (x - x_prev) - (math.pi / 2)
        slope_v = slope_f(x_t) if y_curr > y_prev else slope_f(-x_t)
        y_t = h * (slope_v + 1) / 2
        return y_t
    else:
        return y_prev


def des_firls(num_taps, bands, desired, antisymmetric=False, grid_density=16):
    order = num_taps - 1
    even_order = order % 2 == 0

    fs = []
    D = []

    _f_resp = partial(linear_ramp, bands, desired)
    step = 0.5 / ((num_taps + 1) * grid_density)

    # Experimental, variable grid density
    # slope_to_step_ratio = 0.001
    # fs = []
    # for i, (a, b) in enumerate(zip(bands, bands[1:])):
    #     slope = math.atan(abs((desired[i + 1] - desired[i]) / (b - a)))
    #     slope = math.degrees(slope) / 90
    #     fs.append(np.arange(a, b,
    #                 (step * slope_to_step_ratio) + ((1 - slope) * step * (1.0 - slope_to_step_ratio))))
    # fs = np.concatenate(fs)

    fs = np.arange(0.0, 0.5 + step, step)
    D = np.array([_f_resp(f) for f in fs])

    A = []

    for f in fs:
        if antisymmetric:
            if even_order:
                row = [2 * math.sin(2 * math.pi * f * n)
                       for n in range(1, (order // 2) + 1)]
            else:
                row = [2 * math.sin(2 * math.pi * f * (n - 0.5))
                       for n in range(1, ((order + 1) // 2) + 1)]
        else:  # symmetric
            if even_order:
                row = [1.0] + [2 * math.cos(2 * math.pi * f * n)
                               for n in range(1, (order // 2) + 1)]
            else:
                row = [2 * math.cos(2 * math.pi * f * (n - 0.5))
                       for n in range(1, ((order + 1) // 2) + 1)]
        A.append(row)

    A = np.array(A)

    # plt.pcolor(A, cmap='Greys')
    # plt.show()

    taps = linalg.lstsq(A, D)[0]

    # glue together the calculated half-taps
    if antisymmetric:
        if even_order:
            taps = np.concatenate((np.flip(taps), np.zeros(1), -taps))
        else:
            taps = np.concatenate((np.flip(taps), -taps))
    else:
        if even_order:
            taps = np.concatenate((np.flip(taps[1:]), taps[0:1], taps[1:]))
        else:
            taps = np.concatenate((np.flip(taps), taps))

    assert(len(taps) == num_taps)

    return np.flip(taps), fs, D


def conv_to_remez(bands, desired):

    bs = []
    ds = []

    for i, (d1, d2) in enumerate(zip(desired, desired[1:])):
        if math.isclose(d1, d2):
            bs.extend([bands[i], bands[i+1]])
            ds.append((d1 + d2) / 2)

    return bs, ds


if __name__ == '__main__':

    def deemph_spec():
        def rolloff(f):
            return (math.log10(f) - 3.0) * -20

        points = [(10.0, -5.0), (30.0, 4.0), (100.0, 7.0),
                  (200.0, 12.0), (250.0, 11.5)]

        for f in np.linspace(300, 7000, 200):
            points.append((f, rolloff(f)))

        freqs = []
        gains = []

        for f, g in points:
            freqs.append(f / 12500)
            gains.append(math.pow(10, g / 20))

        return freqs, gains

    NUM_TAPS = 151
    TRANS_WIDTH = 0.01

    data = {
        "Hilbert transformer": (
            [0.0, TRANS_WIDTH, 0.5 - TRANS_WIDTH, 0.5],
            [0.0, 1.0, 1.0, 0.0], True),
        "allpass": (
            [0.0, TRANS_WIDTH, 0.5 - TRANS_WIDTH, 0.5],
            [0.0, 1.0, 1.0, 0.0], False),
        "lowpass": (
            [0.0, 0.25 - TRANS_WIDTH, 0.25, 0.5],
            [1.0, 1.0, 0.0, 0.0], False),
        "highpass": (
            [0.0, 0.25 - TRANS_WIDTH, 0.25, 0.5],
            [0.0, 0.0, 1.0, 1.0], False),
        "bandpass": (
            [0.0, 0.15 - TRANS_WIDTH, 0.15, 0.35 - TRANS_WIDTH, 0.35, 0.5],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0], False),
        "sqw": (
            [0.0, 0.1 - TRANS_WIDTH, 0.1, 0.2 - TRANS_WIDTH, 0.2, 0.3 -
                TRANS_WIDTH, 0.3, 0.4 - TRANS_WIDTH, 0.4, 0.5],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], False),
        "deemph": (
            *deemph_spec(), False)
    }

    for name, (bands, desired, antis) in data.items():
        taps, fs, des = des_firls(NUM_TAPS, bands, desired,    antis)

        bands_remez, des_remez = conv_to_remez(bands, desired)

        if not name in ['allpass', 'deemph']:
            taps_test = np.flip(sig.remez(
                NUM_TAPS, bands_remez, des_remez, type='hilbert' if antis else 'bandpass'))

        fig, axs = plt.subplots(4, figsize=(40, 50))
        fig.tight_layout()
        fig.suptitle(name)

        for ax in axs:
            ax.grid(which='major')
            ax.grid(which='minor', alpha=0.2)

        plot(taps, fs, des, axs, "Least-Squares")
        if not name in ['allpass', 'deemph']:
            plot(taps_test, None, None, axs,
                 "Parks-McClellan (Remez)", color='gray', alpha=0.5)

        for ax in axs:
            ax.legend(loc="upper left", edgecolor="black")

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.95,
                            wspace=0.4,
                            hspace=0.4)
        plt.show()
