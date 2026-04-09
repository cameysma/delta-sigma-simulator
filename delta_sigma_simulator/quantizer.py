import numpy as np

from scipy.optimize import brentq


class Quantizer:
    def next(self, input_signal):
        raise NotImplementedError

    def reset(self):
        self.v = 1.0


class QuantizerDelayHysteresis(Quantizer):
    def __init__(self, t_d, v_s=0.0):
        # Delay
        self.t_d = t_d
        # Threshold
        self.v_s = v_s
        # Output value
        self.v = 1.0
        # Time tolerance for root finding
        self.t_tol = 5e-324
        # Time step for initial root finding
        self.t_step = 10e-12

    def next(self, input_signal):
        y = lambda t: input_signal(t)[0] - self.v_s * self.v

        dt = 0

        while y(dt) * y(0) > 0:
            dt += self.t_step

        dt = self.t_d + brentq(y, dt - self.t_step, dt, xtol=self.t_tol)

        self.v *= -1.0

        return dt


class QuantizerClock(Quantizer):
    def __init__(self, f_c, n=1.0):
        # Clock frequency
        self.f_c = f_c
        # Output value
        self.v = 1.0
        # Variance of the input noise
        self.n = n

    def next(self, input_signal):
        dt = 1 / self.f_c

        self.v = (
            -1.0 if input_signal(dt)[0] < np.sqrt(self.n) * np.random.randn() else +1.0
        )

        return dt
