import numpy as np
import matplotlib.pyplot as plt

from .wave import BinaryWave


class DeltaSigmaModulator:
    def __init__(self, filter, quantizer):
        self.filter = filter
        self.quantizer = quantizer

    def simulate_filter(self, dt, v, n=0):
        dt = np.atleast_1d(dt)

        y = np.zeros_like(dt, dtype=np.float64)

        for ui in self.u:
            y += self.filter(ui).derivative(n)(self.t + dt)

        y += self.filter.step_response(dt, n) * -v

        y += self.filter.natural_response(dt, self.y, n)

        return y

    def step(self, plot=False):
        dt = self.quantizer.next(self.simulate_filter)

        if plot:
            print(
                f"{self.t:.2e} s: dt = {dt:.2e} s, v_prev = {self.quantizer.v_prev:.2f}, v = {self.quantizer.v:.2f}"
            )
            print(f"Filter state: {self.y}")

            t = np.linspace(0, 2 * dt, 1000)
            y = self.simulate_filter(t, self.quantizer.v_prev, 0)
            plt.plot((self.t + t) * 1e9, y, c="k")
            plt.axvline((self.t + dt - self.quantizer.t_d) * 1e9, c="r", ls="--")
            plt.xlabel("Time [ns]")
            plt.ylabel("Filter Output")
            plt.grid()

        # Update state
        self.y = np.array(
            [
                self.filter.step_response(dt, i)[0] * -self.quantizer.v_prev
                + self.filter.natural_response(dt, self.y, i)[0]
                for i in range(len(self.y))
            ]
        )

        self.t += dt

    def reset(self):
        # Reset the current time, filter state, and quantizer state
        self.t = 0.0
        self.y = np.zeros_like(self.filter.p)
        self.quantizer.reset()

        # Adjust the initial filter state to ensure the first quantization event occurs close to t=0
        a = self.simulate_filter(0, self.quantizer.v, 0)[0]
        self.y[0] -= a

    def simulate(self, u, t, run=True, run_until=-1):
        # List of input signals, e.g., SineWave, BinaryWave, or a DC value
        self.u = np.atleast_1d(u)

        # Bring the modulator to the initial state
        self.reset()

        # Edge times of the output binary wave
        e = np.array([0])

        while run:
            self.step()

            e = np.append(e, self.t)

            print(
                f"Progress: {len(e)} {self.t:.2e} s / {t:.2e} s [{self.t / t * 100:.2f}%]",
                end="\r",
            )

            if len(e) == run_until or self.t >= t:
                run = False

        return BinaryWave(e)
