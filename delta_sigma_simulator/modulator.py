import numpy as np
import matplotlib.pyplot as plt

from .wave import BinaryWave


class DeltaSigmaModulator:
    def __init__(self, h_v, q, h_u=None):
        # Feedback filter
        self.h_v = h_v
        # Feedin filter
        self.h_u = h_u if h_u is not None else h_v
        # Quantizer
        self.q = q

    def simulate_filter(self, dt):
        dt = np.atleast_1d(dt)

        y = np.zeros_like(dt, dtype=np.float64)

        for y_u_i in self.y_u:
            y += y_u_i(self.t + dt)

        y -= self.y_v(self.t + dt)

        return y

    def step(self):
        self.t += self.q.next(self.simulate_filter)

        self.y_v.append(self.t)

    def reset(self):
        # Reset the current time, filter state, and quantizer state
        self.t = 0.0
        self.q.reset()

    def simulate(self, u, t=None, n=None, run=True):
        # List of inputs
        self.u = np.atleast_1d(u)
        # Loop filter outputs
        self.y_u = np.array([u_i.filter(self.h_u) for u_i in self.u])
        self.y_v = BinaryWave().filter(self.h_v)

        self.reset()

        while run:
            self.step()

            if t is not None and self.t >= t:
                run = False

            if n is not None and len(self.y_v.e) >= n:
                run = False

        return BinaryWave(self.y_v.e)
