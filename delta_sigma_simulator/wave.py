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
    def __init__(self, e=None, E=1.0, h=None):
        """
        Arguments:
            e: Array of time points where the signal changes value.
            E: Amplitude of the binary wave.
            h: Filter to apply to the binary wave. If None, no filtering is applied.
        """
        if e is None:
            e = np.array([0.0])
        else:
            e = np.atleast_1d(np.asarray(e, dtype=np.float64))

        self.E = E
        self.h = h if h is not None else FilterUnit()

        # Edge times and the state of the filter at the end of each transition, stored
        # in buffers that grow geometrically. Keeping the state avoids having to
        # evaluate the contribution of every transition each time the wave is
        # evaluated: the response after a transition is the step response due to that
        # transition, plus the natural response due to the state at the end of the
        # previous transition.
        self._n = 0
        self._e = np.zeros(max(len(e), 1))
        self._x = np.zeros((len(self._e), self.h.state_size))

        for e_i in e:
            self.append(e_i)

    @property
    def e(self):
        """
        Time points where the signal changes value.
        """
        return self._e[: self._n]

    @property
    def x(self):
        """
        State of the filter at the end of each transition.
        """
        return self._x[: self._n]

    def _grow(self):
        e = np.zeros(2 * len(self._e))
        e[: self._n] = self._e[: self._n]

        x = np.zeros((len(e), self.h.state_size))
        x[: self._n] = self._x[: self._n]

        self._e = e
        self._x = x

    def append(self, e):
        """
        Append a new edge to the binary wave.

        Arguments:
            e: The time point where the signal changes value.
        """
        if self._n > 0 and e < self._e[self._n - 1]:
            raise ValueError("Edges must be appended in chronological order.")

        if self._n == len(self._e):
            self._grow()

        # State due to the transitions up to and including this one
        x = 2 * (-1) ** self._n * self.E * self.h.step_state

        if self._n > 0:
            x = x + self.h.state_propagate(
                e - self._e[self._n - 1], self._x[self._n - 1]
            )

        self._e[self._n] = e
        self._x[self._n] = x

        self._n += 1

    def __truediv__(self, other):
        """
        Divide the binary wave by a scalar.

        Arguments:
            other: The scalar to divide by.
        """
        return BinaryWave(self.e, self.E / other, self.h)

    def delay(self, tau):
        """
        Return a new BinaryWave that is delayed by tau seconds.

        Arguments:
            tau: The amount of time to delay the binary wave, in seconds.
        """
        return BinaryWave(self.e + tau, self.E, self.h)

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
        t = np.asarray(t, dtype=np.float64)

        # Index of the last transition at or before t, or -1 before the first one
        i = np.searchsorted(self.e, t, side="right") - 1

        j = np.maximum(i, 0)

        y = self.h.state_response(np.where(i < 0, 0.0, t - self._e[j]), self._x[j])
        y = np.where(i < 0, 0.0, y)

        # Reference step at t = 0, which sets the initial value of the wave
        y = y - self.E * np.asarray(self.h.step_response(t)).reshape(t.shape)

        return y[()] if y.ndim == 0 else y
