import numpy as np

from scipy.optimize import brentq


class Quantizer:
    def next(self, input_signal):
        raise NotImplementedError

    def reset(self):
        self.v = 1.0


class QuantizerDelayHysteresis(Quantizer):
    def __init__(self, t_d=0.0, v_s=0.0):
        # Current output value
        self.v = +1.0
        # Delay
        self.t_d = t_d
        # Threshold
        self.v_s = v_s
        # Time tolerance for root finding
        self.t_tol = 5e-324
        # Time step for initial root finding
        self.t_step = 10e-12

    def next(self, y):
        y_v = lambda t : self.v * y(t) + self.v_s

        dt = self.t_step

        while y_v(dt) > 0:
            dt += self.t_step

        try:
            t0 = brentq(y_v, dt - self.t_step, dt, xtol=self.t_tol)
        except ValueError:
            t0 = self.t_step

        dt = t0 + self.t_d

        self.v *= -1 

        return dt

class QuantizerClock(Quantizer):
    def __init__(self, f_c, n=1.0):
        # Current output value
        self.v = +1.0
        # Clock frequency
        self.f_c = f_c
        # Variance of the input noise
        self.n = n

    def next(self, y):
        y_v = lambda t : self.v * y(t) + np.random.randn() * np.sqrt(self.n)

        dt = 1 / self.f_c

        while y_v(dt) > 0:
            dt += 1 / self.f_c

        self.v *= -1

        return dt
