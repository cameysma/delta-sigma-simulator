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

        # Precompute the output at each edge- to speed up the __call__ method
        if y is not None:
            self.y = y
        else:
            num_p = len(self.h.p)
            num_e = len(self.e)

            self.y = np.zeros((num_p, num_e))

            for j in range(num_e - 1):
                de = self.e[j + 1] - self.e[j]

                s = (-1) ** j * self.E

                for i in range(num_p):
                    self.y[i, j + 1] += self.h.step_response(de, i) * s
                    self.y[i, j + 1] += self.h.natural_response(de, self.y[:, j], i)

    def append(self, e):
        """
        Append a new edge to the binary wave.

        Arguments:
            e: The time point where the signal changes value.
        """
        num_p = len(self.h.p)

        y = np.zeros((num_p, 1))

        j = len(self.e) - 1

        de = e - self.e[j]

        s = (-1) ** j * self.E

        for i in range(num_p):
            y[i, 0] += self.h.step_response(de, i) * s
            y[i, 0] += self.h.natural_response(de, self.y[:, j], i)

        self.e = np.append(self.e, e)
        self.y = np.append(self.y, y, axis=1)

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
        y = np.zeros_like(t)

        for i, ti in enumerate(t):
            if ti < self.e[0]:
                # If the time point is before the first edge, the output is zero
                continue
            elif ti >= self.e[-1]:
                # If the time point is after the last edge, do not search for the correct edge
                j = len(self.e) - 1
            else:
                # Search for the correct edge using a while loop
                j = 0

                while j != len(self.e) - 1 and not (
                    self.e[j] <= ti and ti < self.e[j + 1]
                ):
                    j += 1

            de = ti - self.e[j]

            s = (-1) ** j * self.E

            y[i] += self.h.step_response(de) * s
            y[i] += self.h.natural_response(de, self.y[:, j])

        return y
