import numpy as np

from delta_sigma_simulator.filter import FilterUnit


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

    def filter(self, h):
        return SineWave(
            self.phasors * h.frequency_response(self.angular_frequencies),
            self.fundamental_frequency,
        )


class BinaryWave:
    def __init__(self, e=None, E=1.0, h=None, y=None):
        """
        Arguments:
            e: Array of time points where the signal changes value.
            E: Amplitude of the binary wave.
            h: Filter to apply to the binary wave. If None, no filtering is applied.
        """
        if e is None:
            self.e = np.array([0.0])
        else:
            self.e = np.atleast_1d(e)

        self.E = E
        self.h = h if h is not None else FilterUnit()

    def append(self, e):
        """
        Append a new edge to the binary wave.

        Arguments:
            e: The time point where the signal changes value.
        """
        self.e = np.append(self.e, e)

    def __truediv__(self, other):
        """
        Divide the binary wave by a scalar.

        Arguments:
            other: The scalar to divide by.
        """
        return BinaryWave(self.e, self.E / other, self.h, self.y / other)

    def delay(self, tau):
        """
        Return a new BinaryWave that is delayed by tau seconds.

        Arguments:
            tau: The amount of time to delay the binary wave, in seconds.
        """
        return BinaryWave(self.e + tau, self.E, self.h, self.y)

    def filter(self, h):
        """
        Return a new BinaryWave that is filtered by the given filter.

        Arguments:
            h: The filter to apply to the binary wave.
        """
        if isinstance(self.h, FilterUnit):
            return BinaryWave(self.e, self.E, h)
        else:
            raise NotImplementedError("Filters cannot be cascaded.")

    def __call__(self, t):
        """
        Evaluate the binary wave at the given time points.

        Arguments:
            t: Array of time points to evaluate the binary wave at.
        """
        y = -self.h.step_response(t)  * self.E

        for i, e_i in enumerate(self.e):
            y += self.h.step_response(t - e_i) * 2 * (-1) ** i * self.E

        return y
