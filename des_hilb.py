import math
import numpy as np

import scipy.signal as sig
from scipy import linalg
from scipy.fftpack import fft, fftshift

import matplotlib.pyplot as plt

from des_firls import des_firls


def plot(a_taps, b_taps, axs, color='blue'):

    axs[1].step(range(len(a_taps)), a_taps,
                where='mid', color=color, marker='.')
    axs[1].set_xlim(-1, len(b_taps))
    # axs[1].set_xlabel('n')
    axs[1].set_ylabel('hn_a')

    axs[0].step(range(len(b_taps)), b_taps,
                where='mid', color=color, marker='.')
    axs[0].set_xlim(-1, len(b_taps))
    # axs[0].set_xlabel('n')
    axs[0].set_ylabel('hn_b')

    A = fft(b_taps, 2048)
    freq = np.linspace(-0.5, 0.5, len(A), endpoint=False)
    A = fftshift(A)
    f_response = 20 * np.log10(np.abs(A))
    p_response = np.unwrap(np.angle(A))

    axs[2].plot(freq, f_response, color=color)
    axs[2].set_xlim(-0.5, 0.5)
    axs[2].set_xticks(np.arange(-0.5, 0.5, 0.02), minor=True)
    yt_min = min(f_response[f_response > -420]) - 10
    axs[2].set_yticks(np.arange(yt_min, 10.0, (10 - yt_min) / 5))
    # axs[2].set_xlabel('Frequency')
    axs[2].set_ylabel('Gain (dB)')

    axs[3].plot(freq, p_response, color=color)
    axs[3].set_xlim(-0.5, 0.5)
    axs[3].set_xticks(np.arange(-0.5, 0.5, 0.02), minor=True)
    axs[3].set_yticks(np.arange(min(p_response), max(
        p_response), (max(p_response) - min(p_response)) / 5))
    # axs[3].set_xlabel('Frequency')
    axs[3].set_ylabel('Phase (radians)')


def des_delay(num_taps):
    delay = np.zeros(num_taps)
    delay[int((num_taps - 1) / 2)] = 1.0

    return delay


def des_parks_mcclellan(num_taps, trans_width):
    taps = sig.remez(
        num_taps, [trans_width, 0.5 - trans_width], [1.0], type='hilbert')

    return des_delay(num_taps), np.flip(taps)


def des_turner(num_taps, trans_width):
    a = trans_width / 2
    omega_1 = a
    omega_2 = 0.5 - a

    def helper_f(x): return math.sin(((a + (2.0 * x)) / a) * math.pi / 4.0)

    def A_f(t):
        if math.isclose(t, 0.0):
            return math.sqrt(2) * (omega_2 - omega_1)
        elif math.isclose(t, math.pi / (2.0 * a)):
            return a * (helper_f(omega_2) - helper_f(omega_1))
        elif math.isclose(t, -math.pi / (2.0 * a)):
            return a * (helper_f(-omega_1) - helper_f(-omega_2))
        else:
            g = (2.0 * math.pi * math.pi * math.cos(a * t)) / \
                (t * ((4.0 * a * a * t * t) - (math.pi * math.pi)))
            return (math.sin((math.pi / 4.0) + (omega_1 * t)) -
                    math.sin((math.pi / 4.0) + (omega_2 * t))) * g

    time = [2 * math.pi * (k - ((num_taps - 1) / 2)) for k in range(num_taps)]
    A = np.array(list(map(A_f, time)))

    return A, np.flip(A)


def des_window(num_taps, trans_width):
    taps = sig.firwin2(num_taps, [0.0, 2 * trans_width, 1.0 - (2 * trans_width), 1.0],
                       [0.0, 1.0, 1.0, 0.0], window=('kaiser', 8), antisymmetric=True)

    return des_delay(num_taps), np.flip(taps)


def des_least_squares_1(num_taps, trans_width):
    assert(num_taps % 2 == 1)

    taps, _, _ = des_firls(num_taps, [0.0, trans_width, 0.5 - trans_width, 0.5],
                           [0.0, 1.0, 1.0, 0.0], True)
    return des_delay(num_taps), taps


def des_least_squares_2(num_taps, trans_width):
    assert(num_taps % 2 == 1)

    i_taps, _, _ = des_firls(num_taps, [0.0, trans_width, 0.5 - trans_width, 0.5],
                             [0.0, 1.0, 1.0, 0.0], False)

    q_taps, _, _ = des_firls(num_taps, [0.0, trans_width, 0.5 - trans_width, 0.5],
                             [0.0, 1.0, 1.0, 0.0], True)

    # just an experiment using the Turner idea
    # those two filters with phase +/-pi/4
    # can be used as an Hilbert transformer
    # (without a delay in the second branch)
    a_taps = (i_taps - q_taps) / math.sqrt(2)
    b_taps = (i_taps + q_taps) / math.sqrt(2)

    return a_taps, b_taps


NUM_TAPS = 151
TRANS_WIDTH = 0.01


methods = {'Parks-McClellan (Remez)': des_parks_mcclellan,
           'Turner': des_turner,
           'Window': des_window,
           'Least Squares π/2': des_least_squares_1,
           'Least Squares π/4': des_least_squares_2}

for name, method in methods.items():
    a_taps, b_taps = method(NUM_TAPS, TRANS_WIDTH)

    # win = sig.windows.kaiser(NUM_TAPS, beta=8)
    # a_taps *= win
    # b_taps *= win

    fig, axs = plt.subplots(5, figsize=(40, 50), gridspec_kw={
        'height_ratios': [1, 1, 1, 1, 4]})
    fig.suptitle(name)
    fig.set_facecolor('#f0f0f0')

    for ax in axs:
        ax.grid(which='major')
        ax.grid(which='minor', alpha=0.2)

    plot(a_taps, b_taps, axs[:-1])

    FS = 1000  # Hz
    G_DELAY = int((NUM_TAPS - 1) / 2)

    t = np.linspace(0, 1 + (G_DELAY / FS), FS + G_DELAY)

    envelope = sig.windows.kaiser(FS + G_DELAY, beta=16)
    test_signal = sig.chirp(t, f0=5, f1=100, t1=1, method='quadratic')
    test_signal *= envelope
    test_signal_a = sig.lfilter(a_taps, [1.0], test_signal)
    test_signal_b = sig.lfilter(b_taps, [1.0], test_signal)
    test_signal_cplx = test_signal_a + (1j * test_signal_b)
    test_signal_amp = np.abs(test_signal_cplx)
    test_signal_freq = np.gradient(
        np.unwrap(np.angle(test_signal_cplx)), t) / (2 * np.pi)

    ax1 = axs[-1]
    ax1.set_xticks(np.arange(min(t), max(t), 0.05))
    ax1.plot(t, test_signal_a, label="A signal")
    ax1.plot(t, test_signal_b, label="B signal")
    ax1.plot(t, test_signal_amp, label="Complex amplitude")
    ax1.plot([], [], label="Frequency", color='maroon')
    ax1.set_xlim(0, 1 + (G_DELAY / FS))
    ax1.set_ylabel('Amplitude')
    plt.legend(loc="upper left", edgecolor="black")

    ax2 = ax1.twinx()
    ax2.plot(t, test_signal_freq, color='maroon')
    ax2.set_ylabel('Frequency (Hz)')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.95,
                        wspace=0.4,
                        hspace=0.4)
    plt.show()
    fig.savefig("plots/plot_{}.png".format(name.lower().replace(' ', '_')
                                           .replace('/', '_').replace('π', 'pi')
                                           .replace('(', '_').replace(')', '_')),
                format="png")
