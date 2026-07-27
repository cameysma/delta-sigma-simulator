import numpy as np

from scipy.optimize import brentq


class Quantizer:
    def next(self, input_signal):
        raise NotImplementedError

    def reset(self):
        self.v = 1.0


class QuantizerDelayHysteresis(Quantizer):
    def __init__(self, t_d=0.0, v_s=0.0):
        # Delay
        self.t_d = t_d
        # Threshold
        self.v_s = v_s
        # Time tolerance for root finding
        self.t_tol = 5e-324
        # Time step for initial root finding
        self.t_step = 10e-12

    def next(self, y):
        if self.v_s != 0.0:
            y_v = lambda t: np.abs(y(t)[0]) - self.v_s
        else:
            y_v = lambda t: y(t)[0]

        dt = 2 * self.t_step

        while y_v(dt) * y_v(dt - self.t_step) > 0:
            dt += self.t_step

        dt = self.t_d + brentq(y_v, dt - self.t_step, dt, xtol=self.t_tol)

        return dt

class QuantizerClock(Quantizer):
    def __init__(self, f_c, n=1.0):
        # Clock frequency
        self.f_c = f_c
        # Variance of the input noise
        self.n = n
        # Current output value
        self.v = +1.0

    def next(self, input_signal):
        y_v = lambda t: input_signal(t)[0] + np.random.randn() * np.sqrt(self.n)

        dt = 1 / self.f_c

        while self.v * y_v(dt) > 0:
            dt += 1 / self.f_c

        self.v *= -1

        return dt
