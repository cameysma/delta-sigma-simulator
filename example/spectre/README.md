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

What limits Spectre here is not its tolerances but how the quantizer has to be
modelled, in two respects.

The first is the time tolerance on the `cross` event, `ttol`, which is what
forces the solver to place a time point on the threshold crossing. It is set to
`1e-9 / f0` throughout; left at its default it raises the floor by some 30 dB
and the extracted intercept point scatters over 7 dB.

The second is that a transient simulation cannot represent an ideal step, so
the quantizer output switches over a finite transition time. The two variants
in `spectre.py`, `SHARP` and `SLOW`, differ in that transition time alone,
`1e-7 / f0` against `1e-3 / f0`, and it costs more than any tolerance does. The
error preset, by contrast, barely matters: running `SHARP` with the default
preset and a relative tolerance of `1e-3` (`spectre.MODERATE`) returns
36.253 dBFS with a spread of 0.001 dB, which is the reference value.

On top of that, the numerical error does not repeat. Every measurement is
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
script.

## Results

Measured with Spectre 25.1 on a single machine, with sixteen periods of the
input to let the demodulation filter settle and sixteen further periods
analysed, for five phases of the input.

At an input amplitude of 0.1, all three simulations return the same
fundamental, -20.00 dBFS, and the sharp edge also returns the same third
harmonic as the event-driven simulator, -138.5 dBFS. With the slow edge that
harmonic stands only 12 dB above the floor and is read 5 dB too high.

| simulator           | state updates | run time | A_oip3 [dBFS] | spread [dB] | floor [dBFS] |
| ------------------- | ------------- | -------- | ------------- | ----------- | ------------ |
| event-driven        | 7.8e3 events  | 7.8 s    | 36.253        | 0.000       | -253         |
| Spectre, t_r = 1e-7 | 5.3e5 steps   | 4.2 s    | 36.251        | 0.005       | -205         |
| Spectre, t_r = 1e-3 | 4.8e5 steps   | 4.0 s    | 34.102        | 2.329       | -146         |

Since the third harmonic falls three times as fast as the input amplitude, it
meets that floor as soon as the input is reduced. The mean stays close to the
correct value for a while, but the spread over the five phases does not, and it
is the spread that shows where each simulator stops being usable. The table
gives the mean, with the spread in brackets.

| U [dBFS] | event-driven  | t_r = 1e-7     | t_r = 1e-3     |
| -------- | ------------- | -------------- | -------------- |
| -10.46   | 35.82 (0.000) | 35.82 (0.000)  | 35.82 (0.078)  |
| -20.00   | 36.25 (0.000) | 36.25 (0.005)  | 34.10 (2.329)  |
| -30.46   | 36.30 (0.000) | 36.31 (0.113)  | 25.14 (7.251)  |
| -40.00   | 36.31 (0.010) | 36.70 (12.667) | 10.61 (9.034)  |
| -50.46   | 36.27 (0.801) | 16.59 (0.737)  | -5.53 (4.208)  |

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
