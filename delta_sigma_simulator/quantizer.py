import numpy as np

from scipy.optimize import brentq


class Quantizer:
    def next(self, input_signal):
        raise NotImplementedError

    def reset(self):
        self.v_prev = 1.0
        self.v = 1.0


class QuantizerDelayHysteresis(Quantizer):
    def __init__(self, t_d, v_s):
        # Delay
        self.t_d = t_d
        # Threshold
        self.v_s = v_s
        # Previous output value
        self.v_prev = 1.0
        # Current output value
        self.v = 1.0
        # Time tolerance for root finding
        self.t_tol = 5e-32
        # Time step for initial root finding
        self.t_step = 10e-12

    def next(self, input_signal):
        y = lambda t: input_signal(t, self.v, 0)[0] - self.v_s * self.v

        dt = 0

        while y(dt) * y(0) > 0:
            dt += self.t_step

        dt = self.t_d + brentq(y, 0, dt, xtol=self.t_tol)

        self.v_prev = self.v
        self.v *= -1.0

        return dt


class QuantizerClock(Quantizer):
    def __init__(self, f_c, n=1.0):
        # Clock frequency
        self.f_c = f_c
        # Previous output value
        self.v_prev = 1.0
        # Current output value
        self.v = 1.0
        # Standard deviation of the input noise
        self.n = n

    def next(self, input_signal):
        dt = 1 / self.f_c

        self.v_prev = self.v
        self.v = (
            -1.0 if input_signal(dt, self.v)[0] < self.n * np.random.randn() else +1.0
        )

        return dt
