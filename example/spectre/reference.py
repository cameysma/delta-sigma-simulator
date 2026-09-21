"""
Run the testbench of `testbench.py` with the event-driven simulator.

The result of this script is the reference against which the Spectre results
are compared: it contains no time step, so its accuracy is limited only by the
precision of the arithmetic.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from delta_sigma_simulator.filter import (  # noqa: E402
    FilterButterworth,
    FilterFirstOrder,
)
from delta_sigma_simulator.modulator import DeltaSigmaModulator  # noqa: E402
from delta_sigma_simulator.quantizer import QuantizerDelayHysteresis  # noqa: E402
from delta_sigma_simulator.wave import SineWave  # noqa: E402

from testbench import (  # noqa: E402
    A_DC,
    DELTA,
    F0,
    FC,
    FG,
    FS,
    FU,
    N_PERIOD,
    N_SETTLE,
    ORDER,
    analyse,
)


def run(uamp=0.1, n_settle=N_SETTLE, n_period=N_PERIOD):
    """Simulate the testbench and analyse the demodulated output."""
    quantizer = QuantizerDelayHysteresis(0.0, DELTA)
    # Bracket the threshold crossings on a tenth of a self-oscillation period
    quantizer.t_step = 0.1 / F0
    modulator = DeltaSigmaModulator(FilterFirstOrder(FG, A_DC), quantizer)

    start = time.perf_counter()

    v = modulator.simulate([SineWave([0, uamp], FU)], t=(n_settle + n_period) / FU)

    t = n_settle / FU + np.arange(0.0, n_period / FU - 0.5 / FS, 1 / FS)
    x = v.filter(FilterButterworth(ORDER, FC))(t)

    elapsed = time.perf_counter() - start

    result = analyse(x, FS, uamp, n_period)
    result["steps"] = len(v.e)
    result["time"] = elapsed
    return result


if __name__ == "__main__":
    r = run(float(sys.argv[1]) if len(sys.argv) > 1 else 0.1)
    print(
        f"transitions={r['steps']} OIP3={r['oip3']:.3f} dBFS "
        f"time={r['time']:.2f} s"
    )
