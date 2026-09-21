"""
Compare the event-driven simulator against Spectre on the same testbench.

Usage: compare.py [spectrum | amplitude | cost | all]

spectrum   the demodulated output spectrum at an input amplitude of 0.1,
           which shows that both simulators return the same harmonics but a
           very different floor in between
amplitude  the extracted third-order intercept point as a function of the
           input amplitude, which shows where each simulator stops resolving
           the third harmonic
cost       the time step, the number of state updates and the run time needed
           by Spectre to reach the reference result at an input amplitude of
           0.01

Each experiment writes a CSV next to this script.
"""

import sys

import numpy as np

import reference
import spectre
from testbench import floor, write_csv

AMPLITUDES = [0.3, 0.1, 0.03, 0.01, 0.003]


def dbfs(u):
    return 20 * np.log10(u)


def spectrum():
    print("spectrum at an input amplitude of 0.1")

    results = {
        "event_driven": reference.run(0.1),
        "conservative": spectre.run("spectrum_conservative", **spectre.CONSERVATIVE),
        "moderate": spectre.run("spectrum_moderate", **spectre.MODERATE),
    }

    for name, r in results.items():
        print(
            f"  {name:>13} OIP3={r['oip3']:7.3f} dBFS floor={floor(r):8.2f} dBFS "
            f"updates={r['steps']:>9} time={r['time']:6.2f} s"
        )

    f = results["event_driven"]["f_over_fu"]
    inside = f <= 4.0001
    write_csv(
        "spectrum.csv",
        {"f_over_fu": f[inside], **{n: r["dbfs"][: len(f)][inside] for n, r in results.items()}},
    )


def amplitude():
    print("third-order intercept point versus input amplitude")

    columns = {"u_dbfs": [], "event_driven": [], "conservative": [], "moderate": []}
    for u in AMPLITUDES:
        runs = {
            "event_driven": reference.run(u),
            "conservative": spectre.run(f"amplitude_conservative_{u}", uamp=u, **spectre.CONSERVATIVE),
            "moderate": spectre.run(f"amplitude_moderate_{u}", uamp=u, **spectre.MODERATE),
        }
        columns["u_dbfs"].append(dbfs(u))
        for name, r in runs.items():
            columns[name].append(r["oip3"])
        print(
            f"  U={dbfs(u):7.2f} dBFS "
            + " ".join(f"{n}={r['oip3']:7.3f}" for n, r in runs.items())
        )

    write_csv("amplitude.csv", columns)


def cost():
    print("cost of resolving the third harmonic at an input amplitude of 0.01")

    r = reference.run(0.01)
    print(
        f"  event-driven      OIP3={r['oip3']:7.3f} dBFS "
        f"updates={r['steps']:>9} time={r['time']:7.2f} s"
    )

    columns = {"mstep": [], "reltol": [], "steps": [], "time": [], "oip3": []}
    for mstep, reltol in [(1e-2, 1e-5), (1e-3, 1e-7), (1e-4, 1e-9)]:
        s = spectre.run(
            f"cost_{mstep}",
            uamp=0.01,
            **dict(spectre.CONSERVATIVE, mstep=mstep, reltol=reltol),
        )
        for name, value in zip(columns, [mstep, reltol, s["steps"], s["time"], s["oip3"]]):
            columns[name].append(value)
        print(
            f"  mstep={mstep:<7g} reltol={reltol:<7g} OIP3={s['oip3']:7.3f} dBFS "
            f"steps={s['steps']:>9} time={s['time']:7.2f} s"
        )

    write_csv("cost.csv", columns)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in ["spectrum", "amplitude", "cost"] if what == "all" else [what]:
        globals()[name]()
