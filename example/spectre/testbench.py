"""
Definition of the testbench that is shared by the two simulators.

An asynchronous sigma-delta modulator with a first-order low-pass loop filter
is driven by a sinusoidal input, its binary output is demodulated by a
Butterworth low-pass filter, and the output-referred third-order intercept
point is read from the discrete Fourier transform of the demodulated output.

Everything is normalized to the self-oscillation frequency, f0 = 1 Hz.
"""

import cmath
import math

import numpy as np

F0 = 1.0  # self-oscillation frequency
BETA0 = 1e-2  # corner frequency of the loop filter, relative to f0
A_DC = 1.0  # DC gain of the loop filter
K = 123  # ratio of the self-oscillation frequency to the input frequency

ORDER = 10  # order of the demodulation filter
N_SETTLE = 16  # periods of the input discarded while the filter settles
N_PERIOD = 16  # periods of the input that are analysed

FU = F0 / K  # input frequency
FG = BETA0 * F0  # corner frequency of the loop filter
FC = 4 * FU  # corner frequency of the demodulation filter
FS = 16 * FC  # rate at which the demodulated output is sampled

# Hysteresis for which the modulator self-oscillates at exactly f0
DELTA = A_DC * math.tanh(math.pi * BETA0 / 2)

# The loop filter is realized as an RC section with a DC gain of one
RR = 1.0
CC = 1 / (2 * math.pi * FG * RR)


def butterworth_poles(f_c=FC, order=ORDER):
    """Poles of a unity-gain Butterworth low-pass filter."""
    w_c = 2 * math.pi * f_c
    return [
        w_c * cmath.exp(1j * math.pi * (2 * k + 1 + order) / (2 * order))
        for k in range(order)
    ]


def butterworth_response(f, f_c=FC, order=ORDER):
    """Frequency response of that filter, evaluated at the frequencies f."""
    p = np.asarray(butterworth_poles(f_c, order))
    s = 2j * np.pi * np.atleast_1d(np.asarray(f, dtype=float))
    return np.prod(-p / (s[:, None] - p[None, :]), axis=1)


def analyse(x, f_s, uamp, n_period=N_PERIOD):
    """
    Analyse a uniformly sampled record of the demodulated output.

    The record must span exactly n_period periods of the input, so that the
    fundamental and the third harmonic fall in bins n_period and 3 n_period.
    The response of the demodulation filter is divided out, so that the
    returned spectrum refers to the output of the modulator itself.
    """
    x = np.asarray(x)
    V = np.fft.rfft(x) / (len(x) / 2)
    f = np.fft.rfftfreq(len(x), 1 / f_s)

    V = np.abs(V / butterworth_response(f))

    v_1 = V[n_period] / uamp
    v_3 = 4 * V[3 * n_period] / uamp**3

    return {
        "f_over_fu": f / FU,
        "dbfs": 20 * np.log10(np.maximum(V, 1e-300)),
        "a_1": V[n_period],
        "a_3": V[3 * n_period],
        "oip3": 20 * np.log10(math.sqrt(2) * abs(v_1**3 / v_3) ** 0.5),
    }


def floor(result, f_max=4.0):
    """Median level in between the harmonics, in dBFS."""
    f, db = result["f_over_fu"], result["dbfs"]
    between = (f > 0.2) & (f < f_max) & (np.abs(f - np.round(f)) > 0.1)
    return float(np.median(db[between]))


def write_csv(path, columns):
    """Write a dictionary of equally long columns to a CSV file."""
    names = list(columns)
    data = np.column_stack([np.asarray(columns[n], dtype=float) for n in names])
    np.savetxt(path, data, delimiter=",", header=",".join(names), comments="")
