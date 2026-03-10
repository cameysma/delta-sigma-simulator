import numpy as np
import matplotlib.pyplot as plt


class SineWave:
    def __init__(self, phasors, fundamental_frequency):
        self.phasors = np.atleast_1d(phasors)
        self.fundamental_frequency = fundamental_frequency

    @property
    def angular_frequencies(self):
        return 2 * np.pi * np.arange(len(self.phasors)) * self.fundamental_frequency

    def __call__(self, t):
        return np.sum(
            self.phasors[:, None]
            * np.exp(1.0j * self.angular_frequencies[:, None] * t),
            axis=0,
        ).real

    def derivative(self, n=1):
        return SineWave(
            self.phasors * (1.0j * self.angular_frequencies) ** n,
            self.fundamental_frequency,
        )

    def plot(self, osr=32):
        fs = 2 * len(self.phasors) * self.fundamental_frequency * osr

        t = np.arange(0, 1 / self.fundamental_frequency, 1 / fs)

        plt.plot(t, self(t))
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")


class BinaryWave:
    def __init__(self, edges, sign=1.0):
        self.edges = edges
        self.sign = sign

    def __call__(self, t):
        y = np.ones_like(t) * self.sign

        for edge in self.edges:
            y[t >= edge] *= -1.0

        return y

    def plot(self):
        t = np.repeat(self.edges, 2)
        y = np.zeros_like(t)
        y[0::4] = -self.sign
        y[1::4] = +self.sign
        y[2::4] = +self.sign
        y[3::4] = -self.sign

        plt.plot(t, y)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")


class FilteredBinaryWave(BinaryWave):
    def __init__(self, edges, filter, sign=1.0, n=0):
        super().__init__(edges, sign)
        self.filter = filter
        self.n = n

    def derivative(self, n=1):
        return FilteredBinaryWave(self.edges, self.filter, self.sign, self.n + n)

    def __call__(self, t):
        y = np.zeros_like(t)

        for i, edge in enumerate(self.edges):
            if i == 0:

                y[t >= edge] += (
                    1
                    * self.filter.step_response(t[t >= edge] - edge, self.n)
                    * self.sign
                )
            else:
                y[t >= edge] += (
                    2
                    * self.filter.step_response(t[t >= edge] - edge, self.n)
                    * self.sign
                    * (-1.0) ** i
                )

        return y
