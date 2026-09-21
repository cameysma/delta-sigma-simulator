"""
Run the testbench of `testbench.py` in Spectre.

The netlist `asdm.scs` is filled in with the requested transient settings,
Spectre is invoked on it, and the strobed output is analysed in the same way as
the output of the event-driven simulator.

The `spectre` executable must be on the PATH, or its location must be given in
the environment variable SPECTRE.
"""

import os
import re
import subprocess
import time

import numpy as np

from testbench import (
    CC,
    DELTA,
    FS,
    FU,
    N_PERIOD,
    N_SETTLE,
    RR,
    analyse,
    butterworth_poles,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SPECTRE = os.environ.get("SPECTRE", "spectre")

# Settings of a typical transient simulation, and a tightened set
MODERATE = {"errpreset": "moderate", "reltol": 1e-3, "mstep": 1e-2, "trise": 1e-3}
CONSERVATIVE = {"errpreset": "conservative", "reltol": 1e-5, "mstep": 1e-2, "trise": 1e-7}


def read_psfascii(path, signal):
    """Read one signal and the time vector from a psfascii transient result."""
    t, v = [], []
    with open(path) as f:
        in_value = False
        for line in f:
            if not in_value:
                in_value = line.startswith("VALUE")
                continue
            if line.startswith("END"):
                break
            name, _, value = line.partition(" ")
            if name == '"time"':
                t.append(float(value))
            elif name == f'"{signal}"':
                v.append(float(value))
    return np.asarray(t), np.asarray(v)


def netlist(uamp, n_settle, n_period, settings):
    """Fill in the netlist template with the requested settings."""
    with open(os.path.join(HERE, "asdm.scs")) as f:
        template = f.read()

    poles = "[" + " ".join(
        f"{z.real:.10g} {z.imag:.10g}" for z in butterworth_poles()
    ) + "]"

    return template.format(
        fu=FU,
        uamp=uamp,
        rr=RR,
        cc=CC,
        delta=DELTA,
        poles=poles,
        tstop=(n_settle + n_period) / FU,
        strobe=1 / FS,
        **settings,
    )


def run(tag, uamp=0.1, n_settle=N_SETTLE, n_period=N_PERIOD, **settings):
    """
    Simulate the testbench in Spectre and analyse the demodulated output.

    The keyword arguments override the entries of MODERATE, which are the
    defaults of the simulator.
    """
    p = dict(MODERATE, **settings)

    path = os.path.join(HERE, f"asdm_{tag}.scs")
    raw = os.path.join(HERE, f"raw_{tag}")
    log = os.path.join(HERE, f"log_{tag}.txt")

    with open(path, "w") as f:
        f.write(netlist(uamp, n_settle, n_period, p))

    start = time.time()
    with open(log, "w") as f:
        status = subprocess.run(
            [SPECTRE, "-format", "psfascii", "-raw", raw, path],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=HERE,
        ).returncode
    wall = time.time() - start

    if status != 0:
        raise RuntimeError(f"spectre failed on {path}, see {log}")

    with open(log) as f:
        text = f.read()
    cpu = re.search(r"Total time required for tran analysis.*?CPU = ([\d.]+) s", text)
    steps = re.search(r"Number of accepted tran steps\s*=\s*(\d+)", text)

    t, v = read_psfascii(os.path.join(raw, "tran.tran.tran"), "vd")
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1e-6):
        raise RuntimeError("the output is not uniformly strobed")

    n = int(round(n_period / FU * FS))
    result = analyse(v[-n:], FS, uamp, n_period)
    result["settings"] = p
    result["steps"] = int(steps.group(1)) if steps else None
    result["time"] = float(cpu.group(1)) if cpu else wall
    return result
