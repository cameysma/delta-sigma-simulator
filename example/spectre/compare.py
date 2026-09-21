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
           by Spectre to approach the reference result at an input amplitude
           of 0.01

Every measurement is repeated for each of the input phases in
`testbench.PHASES`. The event-driven simulator returns the same result for all
of them, while the numerical error of the transient solver does not repeat, so
the experiments report the spread over those repetitions as well as the mean.

The two Spectre variants differ only in the transition time of the quantizer
output, which a transient simulation cannot make zero.

Each experiment writes a CSV next to this script.
"""

import sys

import numpy as np

import reference
import spectre
from testbench import PHASES, floor, write_csv

AMPLITUDES = [0.3, 0.1, 0.03, 0.01, 0.003]

FLOWS = {
    "event_driven": lambda tag, **kw: reference.run(**kw),
    "sharp_edge": lambda tag, **kw: spectre.run(tag, **dict(spectre.SHARP, **kw)),
    "slow_edge": lambda tag, **kw: spectre.run(tag, **dict(spectre.SLOW, **kw)),
}


def dbfs(u):
    return 20 * np.log10(u)


def repeat(flow, tag, **kw):
    """Run one flow once for every phase of `testbench.PHASES`."""
    return [FLOWS[flow](f"{tag}_{i}", phase=p, **kw) for i, p in enumerate(PHASES)]


def report(name, runs, key=lambda r: r["oip3"]):
    values = np.array([key(r) for r in runs])
    print(
        f"  {name:>13} mean={values.mean():8.3f} min={values.min():8.3f} "
        f"max={values.max():8.3f} spread={np.ptp(values):7.3f}"
    )
    return values


def spectrum():
    print("spectrum at an input amplitude of 0.1")

    columns = {}
    for flow in FLOWS:
        runs = repeat(flow, f"spectrum_{flow}", uamp=0.1)
        report(flow, runs)
        report(flow + " floor", runs, floor)
        print(
            f"  {flow:>13} updates={runs[0]['steps']:>9} "
            f"time={np.mean([r['time'] for r in runs]):6.2f} s"
        )
        # The spectra of the repetitions differ only in their floor, so the
        # first one is representative
        columns[flow] = runs[0]["dbfs"]
        frequency = runs[0]["f_over_fu"]

    # Keep the band up to four times the input frequency, where the
    # demodulation filter has not yet attenuated the output
    inside = frequency <= 4.0001
    write_csv(
        "spectrum.csv",
        {"f_over_fu": frequency[inside], **{n: v[inside] for n, v in columns.items()}},
    )


def amplitude():
    print("third-order intercept point versus input amplitude")

    columns = {"u_dbfs": [dbfs(u) for u in AMPLITUDES]}
    for flow in FLOWS:
        mean, low, high = [], [], []
        for u in AMPLITUDES:
            values = report(
                f"{flow} U={dbfs(u):.1f}",
                repeat(flow, f"amplitude_{flow}_{u}", uamp=u),
            )
            mean.append(values.mean())
            low.append(values.min())
            high.append(values.max())
        columns[flow] = mean
        columns[flow + "_min"] = low
        columns[flow + "_max"] = high

    write_csv("amplitude.csv", columns)


def cost():
    print("cost of approaching the reference result at an input amplitude of 0.01")

    runs = repeat("event_driven", "cost_reference", uamp=0.01)
    exact = report("event-driven", runs)
    print(
        f"  {'event-driven':>13} updates={runs[0]['steps']:>9} "
        f"time={np.mean([r['time'] for r in runs]):7.2f} s"
    )

    columns = {"mstep": [], "steps": [], "time": [], "oip3": [], "error": []}
    for mstep, reltol in [(1e-2, 1e-5), (1e-3, 1e-7), (1e-4, 1e-9)]:
        runs = repeat(
            "sharp_edge", f"cost_{mstep}", uamp=0.01, mstep=mstep, reltol=reltol
        )
        values = report(f"mstep={mstep:g}", runs)
        error = np.abs(values - exact.mean())
        for name, value in zip(
            columns,
            [
                mstep,
                runs[0]["steps"],
                np.mean([r["time"] for r in runs]),
                values.mean(),
                error.max(),
            ],
        ):
            columns[name].append(value)
        print(
            f"  {'':>13} steps={runs[0]['steps']:>9} "
            f"time={np.mean([r['time'] for r in runs]):7.2f} s "
            f"worst error={error.max():6.3f} dB"
        )

    write_csv("cost.csv", columns)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in ["spectrum", "amplitude", "cost"] if what == "all" else [what]:
        globals()[name]()
