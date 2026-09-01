import numpy as np
from numpy.polynomial import Polynomial


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

        # Filter of the step response and the state machinery derived from it, both
        # evaluated on first use to keep the constructor cheap
        self._step_filter = None
        self._step_state = None
        self._state_basis = None

    def __call__(self, u):
        return u.filter(self)

    def frequency_response(self, omega):
        s = np.atleast_1d(1.0j * omega)

        num_eval = self.num(s)
        den_eval = self.den(s)

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.isclose(den_eval, 0), 1e9, self.k * num_eval / den_eval)

    def impulse_response(self, t, n=0):
        # The response is zero before t = 0, so the exponentials are only evaluated for
        # t >= 0, where they are bounded for stable poles
        y = np.sum(
            self.residues[:, None]
            * self.p[:, None] ** n
            * np.exp(self.p[:, None] * np.maximum(t, 0.0)),
            axis=0,
        ).real

        y[t < 0] = 0.0

        return y[0] if y.size == 1 else y

    @property
    def step_filter(self):
        """
        Filter whose impulse response equals the step response of this filter, that is
        G(s)/s, obtained by adding a pole at the origin.
        """
        if self._step_filter is None:
            self._step_filter = Filter(self.z, np.concatenate(([0.0], self.p)), self.k)

        return self._step_filter

    def step_response(self, t, n=0):
        return self.step_filter.impulse_response(t, n)

    @property
    def state_size(self):
        """
        Number of state variables. The response to a piecewise constant input is
        described by its value and its first len(p) derivatives.
        """
        return len(self.p) + 1

    @property
    def state_basis(self):
        """
        Poles p of the step response, together with the scale s and the inverse of the
        Vandermonde matrix V, with V[k, m] = (p[m] / s) ** k, used to expand a state
        into the modal coefficients of the response.

        The poles are scaled by the largest pole magnitude, so that a state whose k-th
        derivative is of the order of s ** k results in a well-conditioned system.
        """
        if self._state_basis is None:
            p = self.step_filter.p

            s = np.max(np.abs(p)) if np.any(p) else 1.0

            v = np.vander(p / s, increasing=True).T

            self._state_basis = (
                p,
                s ** np.arange(self.state_size),
                v,
                np.linalg.inv(v),
            )

        return self._state_basis

    @property
    def step_state(self):
        """
        State increment due to a unit step applied at t = 0.
        """
        if self._step_state is None:
            self._step_state = np.array(
                [self.step_response(0.0, n) for n in range(self.state_size)]
            )

        return self._step_state

    def state_coefficients(self, x):
        """
        Coefficients c of the modal expansion of the response, such that the k-th
        derivative of the response at t = 0 satisfies sum_m c_m p_m^k = x_k.

        Arguments:
            x: State, of shape (..., state_size).
        """
        _, s, _, v_inv = self.state_basis

        return (np.asarray(x, dtype=np.float64) / s) @ v_inv.T

    def state_response(self, t, x, n=0):
        """
        Evaluate the n-th derivative of the response to a constant input at t >= 0,
        given the state x at t = 0.

        Arguments:
            t: Time points to evaluate the response at.
            x: State at t = 0, of shape (..., state_size), broadcast against t.
            n: Order of the derivative.
        """
        p, _, _, _ = self.state_basis

        t = np.asarray(t, dtype=np.float64)

        c = self.state_coefficients(x)

        return np.sum(c * p**n * np.exp(p * t[..., None]), axis=-1).real

    def state_propagate(self, t, x):
        """
        Return the state at t >= 0 of the response to a constant input, given the state
        x at t = 0.

        Arguments:
            t: Time to propagate the state over.
            x: State at t = 0, of shape (..., state_size).
        """
        p, s, v, _ = self.state_basis

        c = self.state_coefficients(x) * np.exp(p * np.asarray(t)[..., None])

        return (c @ v.T).real * s


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

        return y[0] if y.size == 1 else y

    def state_response(self, t, x, n=0):
        # Override state response to avoid repeated pole at zero: a constant input
        # results in a ramp, described by the state x = [y(0), y'(0)]
        t = np.asarray(t, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)

        if n == 0:
            return x[..., 0] + x[..., 1] * t
        elif n == 1:
            return np.broadcast_to(x[..., 1], np.broadcast(x[..., 1], t).shape).copy()
        else:
            return np.zeros(np.broadcast(x[..., 1], t).shape)

    def state_propagate(self, t, x):
        return np.stack(
            [self.state_response(t, x, 0), self.state_response(t, x, 1)], axis=-1
        )


class FilterUnit(Filter):
    def __init__(self):
        super().__init__(np.array([]), np.array([]), 1.0)
