import numpy as np
import matplotlib.pyplot as plt

from .wave import BinaryWave


class DeltaSigmaModulator:
    def __init__(self, h, q):
        self.h = h
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

    def reset(self):
        # Reset the current time, filter state, and quantizer state
        self.t = 0.0
        self.q.reset()

        # Adjust the initial filter state to ensure the first quantization event occurs close to t=0
        a = self.simulate_filter(0)[0]

        self.y_v.y[0, 0] -= a

    def simulate(self, u, t, run=True):
        # List of inputs
        self.u = np.atleast_1d(u)
        # Loop filter outputs
        self.y_u = np.array([u_i.filter(self.h) for u_i in self.u])
        self.y_v = BinaryWave([0]).filter(self.h)

        self.reset()

        while run:
            self.step()

            # This also updates the corresponding filter output
            self.y_v.append(self.t)

            if self.t >= t:
                run = False

        return BinaryWave(self.y_v.e)
