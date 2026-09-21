# Comparison with a circuit simulator

This testbench compares the event-driven simulator of this repository against
Spectre, on one and the same asynchronous sigma-delta modulator.

The modulator has a first-order low-pass loop filter with `beta0 = 1e-2` and a
DC gain of one, and a quantizer without delay whose hysteresis is set to
`tanh(pi beta0 / 2)`, so that it self-oscillates at `f0`. Everything is
normalized to `f0 = 1 Hz`. The input is a sinusoid at `f0 / 123`, the binary
output is demodulated by a tenth-order Butterworth low-pass filter at four
times the input frequency, and the output-referred third-order intercept point
is read from the discrete Fourier transform of the demodulated output.

The point of the comparison is that this intercept point has to be extracted
from a third harmonic that lies almost 120 dB below the fundamental. A circuit
simulator solves the modulator by numerical integration on a discrete time
grid, and the local truncation error of every time step ends up in the output
as a floor. The event-driven simulator has no time step at all: it represents
the binary output by its transition times, which it determines to machine
precision.

## Files

| file           | purpose                                                       |
| -------------- | ------------------------------------------------------------- |
| `testbench.py` | the testbench parameters and the analysis shared by both flows |
| `reference.py` | runs the testbench with the event-driven simulator             |
| `spectre.py`   | runs the testbench in Spectre                                  |
| `compare.py`   | the three experiments below, writing a CSV each                |
| `asdm.scs`     | the Spectre netlist, with the transient settings left open     |
| `quantizer.va` | the quantizer, in Verilog-A                                    |

In the netlist, the threshold crossings are detected with a `cross` event, and
the demodulation filter is part of the netlist as a Laplace element, so that
the binary output is filtered before it is strobed and out-of-band content
cannot alias into the band of interest. The modulator has no stable operating
point, so the transient starts from an initial condition rather than from a DC
solution.

Two details of the Spectre side turn out to matter as much as the transient
tolerances themselves.

The first is the time tolerance on the `cross` event, `ttol`, which is what
forces the solver to place a time point on the threshold crossing. At the
default transient settings it dominates the result: with `ttol` at `1e-9 / f0`
the floor is -172 dBFS and the intercept point is within 0.3 dB of the
reference, while with a loose tolerance the floor rises to -144 dBFS and the
intercept point scatters over 7 dB.

The second is that the numerical error does not repeat. Every measurement is
therefore run once for each phase in `testbench.PHASES`, which leaves the
modulator itself untouched but gives the solver a different error pattern. The
event-driven simulator returns the same result for all of them, to every digit;
the Spectre results scatter, and the scatter grows as the third harmonic
approaches the floor.

## Running

The event-driven side needs nothing but this repository.

```
python reference.py 0.1
```

The Spectre side needs `spectre` on the PATH, or its location in the
environment variable `SPECTRE`, and a licence. Then

```
python compare.py all
```

runs all three experiments, or name one of `spectrum`, `amplitude` and `cost`
to run only that one. Each experiment writes its results to a CSV next to the
script. Two sets of transient settings are used throughout: `MODERATE`, which
are the defaults of the simulator with a maximum time step of one percent of
the self-oscillation period, and `CONSERVATIVE`, which tightens the error
preset and the relative tolerance and makes the output transition of the
quantizer essentially instantaneous.

## Results

Measured with Spectre 25.1 on a single machine, with sixteen periods of the
input to let the demodulation filter settle and sixteen further periods
analysed, for five phases of the input.

At an input amplitude of 0.1, all three simulations return the same
fundamental, -20.00 dBFS, and the same third harmonic, -138.5 dBFS. They
differ in the floor in between, and in whether they return the same answer
twice.

| simulator            | state updates | run time | A_oip3 [dBFS] | spread [dB] | floor [dBFS] |
| -------------------- | ------------- | -------- | ------------- | ----------- | ------------ |
| event-driven         | 7.8e3 events  | 7.6 s    | 36.253        | 0.000       | -253         |
| Spectre conservative | 5.3e5 steps   | 4.2 s    | 36.251        | 0.005       | -205         |
| Spectre moderate     | 4.2e5 steps   | 3.3 s    | 36.157        | 0.255       | -172         |

Since the third harmonic falls three times as fast as the input amplitude, it
meets that floor as soon as the input is reduced. The mean stays close to the
correct value for a while, but the spread over the five phases does not, and it
is the spread that shows where each simulator stops being usable. The table
gives the mean, with the spread in brackets.

| U [dBFS] | event-driven  | conservative   | moderate      |
| -------- | ------------- | -------------- | ------------- |
| -10.46   | 35.82 (0.000) | 35.82 (0.000)  | 35.80 (0.008) |
| -20.00   | 36.25 (0.000) | 36.25 (0.005)  | 36.16 (0.255) |
| -30.46   | 36.30 (0.000) | 36.31 (0.113)  | 34.85 (5.880) |
| -40.00   | 36.31 (0.010) | 36.70 (12.667) | 25.31 (9.046) |
| -50.46   | 36.27 (0.801) | 16.59 (0.737)  | 10.99 (7.241) |

The remedy in a circuit simulator is a smaller time step. At an input amplitude
of 0.01, where the reference value is 36.31 dBFS, it takes about a thousand
time steps per self-oscillation period to get within 0.1 dB. Reducing the time
step by another decade makes the result worse rather than better, since the
round-off accumulated over so many steps then takes over.

| simulator    | maximum time step | state updates | run time | worst error |
| ------------ | ----------------- | ------------- | -------- | ----------- |
| Spectre      | 1e-2 / f0         | 5.3e5         | 4.2 s    | 8.40 dB     |
| Spectre      | 1e-3 / f0         | 4.0e6         | 30 s     | 0.08 dB     |
| Spectre      | 1e-4 / f0         | 3.9e7         | 290 s    | 0.30 dB     |
| event-driven | ---               | 7.9e3         | 7.5 s    | 0.01 dB     |
