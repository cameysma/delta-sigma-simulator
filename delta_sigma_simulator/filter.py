import numpy as np
from numpy.polynomial import Polynomial

from .wave import SineWave, BinaryWave, FilteredBinaryWave


class Filter:
    def __init__(self, z, p, k):
        self.z = z
        self.p = p
        self.k = k

        self.num = (
            Polynomial.fromroots(self.z) if len(self.z) > 0 else Polynomial([1.0])
        )
        self.den = (
            Polynomial.fromroots(self.p) if len(self.p) > 0 else Polynomial([1.0])
        )

        self.residues = self.k * self.num(self.p) / self.den.deriv()(self.p)

        if len(p) != len(np.unique(p)):
            raise NotImplementedError("Repeated p are not supported.")

        if len(z) > len(p):
            raise NotImplementedError("Improper transfer functions are not supported.")

    def __call__(self, u):
        if isinstance(u, SineWave):
            return SineWave(
                self.frequency_response(u.angular_frequencies) * u.phasors,
                u.fundamental_frequency,
            )
        if isinstance(u, BinaryWave):
            return FilteredBinaryWave(u.edges, self, u.sign)
        else:
            raise NotImplementedError("Only SineWave and BinaryWave are supported.")

    def frequency_response(self, omega):
        omega = np.atleast_1d(omega)
        s = 1.0j * omega

        num_eval = self.num(s)
        den_eval = self.den(s)

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.isclose(den_eval, 0), 1e9, self.k * num_eval / den_eval)

    def impulse_response(self, t, n=0):
        y = np.sum(
            self.residues[:, None] * self.p[:, None] ** n * np.exp(self.p[:, None] * t),
            axis=0,
        ).real

        y[t < 0] = 0.0

        return y

    def step_response(self, t, n=0):
        return Filter(self.z, np.concatenate(([0], self.p)), self.k).impulse_response(
            t, n
        )

    def natural_response(self, t, y, n=0):
        c = np.linalg.solve(np.vander(self.p, increasing=True).T, y)

        y = np.sum(
            c[:, None] * self.p[:, None] ** n * np.exp(self.p[:, None] * t),
            axis=0,
        ).real

        y[t < 0] = 0.0

        return y


class FilterFirstOrder(Filter):
    def __init__(self, f_p, a=1.0):
        z = np.array([])
        p = np.array([-2 * np.pi * f_p])
        k = 2 * np.pi * f_p * a

        super().__init__(z, p, k)


class FilterSecondOrder(Filter):
    def __init__(self, f_z, f_p1, f_p2, a=1.0):
        z = np.array([-2 * np.pi * f_z])
        p = np.array([-2 * np.pi * f_p1, -2 * np.pi * f_p2])
        k = 2 * np.pi * f_p1 * f_p2 / f_z * a

        super().__init__(z, p, k)


class FilterButterworth(Filter):
    def __init__(self, n, f):
        z = np.array([])
        p = (
            2
            * np.pi
            * f
            * np.exp(1j * np.pi * (np.arange(n) + 0.5) / n + 1j * np.pi / 2)
        )
        k = (2 * np.pi * f) ** n

        super().__init__(z, p, k)


class FilterIntegrator(Filter):
    def __init__(self, k):
        z = np.array([])
        p = np.array([0.0])

        super().__init__(z, p, k)

    def step_response(self, t, n=0):
        # Override step response to avoid repeated pole at zero
        t = np.atleast_1d(t)

        if n == 0:
            y = self.k * t
        elif n == 1:
            y = self.k * np.ones_like(t)
        else:
            y = self.k * np.zeros_like(t)

        y[t < 0] = 0.0

        return y

    def natural_response(self, t, y, n=0):
        t = np.atleast_1d(t)

        if n == 0:
            y = y[0] * np.ones_like(t)
        else:
            y = y[0] * np.zeros_like(t)

        y[t < 0] = 0.0

        return y
