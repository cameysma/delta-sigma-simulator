import numpy as np

from scipy.optimize import brentq


class Quantizer:
    def next(self, y, dt):
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
        self.t_tol = 5e-324

    def next(self, simulate_filter, dt):
        dt = self.t_d + brentq(lambda t: simulate_filter(t, self.v)[0] - self.v_s * self.v, 0, 2 * dt, xtol=self.t_tol)

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

    def next(self, simulate_filter, dt=0.0):
        dt = 1 / self.f_c

        self.v_prev = self.v
        self.v = -1.0 if simulate_filter(dt, self.v)[0] < self.n * np.random.randn() else +1.0

        return dt
