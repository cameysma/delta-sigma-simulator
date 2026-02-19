import numpy as np
from numpy.polynomial import Polynomial


class DeltaSigmaModulator:
    def __init__(self, filter, quantizer) -> None:
        self.filter = filter
        self.quantizer = quantizer

    def simulate_filter(self, dt, v, n=0):
        dt = np.atleast_1d(dt)
        y = np.zeros_like(dt, dtype=np.float64)

        for i in range(len(self.u)):
            y += self.filter.sinusoidal_response(self.f * i, self.t + dt, n) * self.u[i]

        y += self.filter.step_response(dt, n) * -v

        y += self.filter.natural_response(dt, self.y, n)

        return y

    def simulate(self, u, f, t, filter=None, run=True, pwl_filename=None, pwl_tt=1e-12):
        # Input signal u[0] + u[1] * cos(2 * pi * f * t) + ...
        self.u = np.atleast_1d(u)
        self.f = f

        # Current time
        self.t = 0.0
        # Initial condition of the filter output
        self.y = np.zeros_like(self.filter.p)
        self.quantizer.reset()

        # Estimate quantization time
        a = self.simulate_filter(0, self.quantizer.v, 0)[0]

        # Force zero-crossing
        self.y[0] -= a + 1e-6

        # Simulation output
        if filter is not None:
            # Filtered output
            w = -np.ones_like(t)
        else:
            # Error
            y = np.zeros_like(t)
            # Unfiltered output
            v = np.zeros_like(t)
            # Edge times
            e = np.array([])

        # Piecewise linear output
        pwl_file = open(pwl_filename, "w") if pwl_filename is not None else None

        while run:
            # Estimate quantization time
            dt = 0

            while np.sign(self.simulate_filter(2 * dt, self.quantizer.v, 0)) == np.sign(self.simulate_filter(0, self.quantizer.v, 0)):
                a = self.simulate_filter(dt, self.quantizer.v, 0)
                b = self.simulate_filter(dt, self.quantizer.v, 1)

                dt += -a / b

            # Find exact quantization time
            dt = self.quantizer.next(self.simulate_filter, dt)

            # Save simulation output
            t_mask = (self.t <= t) & (t < self.t + dt)

            if filter is None:
                y[t_mask] = self.simulate_filter(
                    t[t_mask] - self.t, self.quantizer.v_prev
                )
                v[t_mask] = self.quantizer.v_prev

                e = np.append(e, self.t)
            else:
                w[self.t <= t] += (
                    2
                    * filter.step_response(t[self.t <= t] - self.t)
                    * self.quantizer.v_prev
                )

            # Update state
            self.y = np.array(
                [
                    self.filter.step_response(dt, i)[0] * -self.quantizer.v_prev
                    + self.filter.natural_response(dt, self.y, i)[0]
                    for i in range(len(self.y))
                ]
            )

            self.t += dt

            if self.t >= t[-1]:
                run = False

            if pwl_file is not None:
                pwl_file.write(
                    f"{self.t - pwl_tt / 2:.12e} {self.quantizer.v_prev:.12e}\n"
                )
                pwl_file.write(f"{self.t + pwl_tt / 2:.12e} {self.quantizer.v:.12e}\n")

        if filter is None:
            return y, v, e
        else:
            return w
