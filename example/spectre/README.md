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

| file            | purpose                                                       |
| --------------- | ------------------------------------------------------------- |
| `testbench.py`  | the testbench parameters and the analysis shared by both flows |
| `reference.py`  | runs the testbench with the event-driven simulator             |
| `spectre.py`    | runs the testbench in Spectre                                  |
| `compare.py`    | the three experiments below, writing a CSV each                |
| `asdm.scs`      | the Spectre netlist, with the transient settings left open     |
| `quantizer.va`  | the quantizer, in Verilog-A                                    |

In the netlist, the threshold crossings are detected with a `cross` event, and
the demodulation filter is part of the netlist as a Laplace element, so that
the binary output is filtered before it is strobed and out-of-band content
cannot alias into the band of interest. The modulator has no stable operating
point, so the transient starts from an initial condition rather than from a DC
solution.

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
analysed.

At an input amplitude of 0.1, all three simulations return the same
fundamental, −20.00 dBFS, and the same third harmonic, −138.5 dBFS. They
differ only in the floor in between.

| simulator            | state updates | run time | A_oip3 [dBFS] | floor [dBFS] |
| -------------------- | ------------- | -------- | ------------- | ------------ |
| event-driven         | 7 833 events  | 7.4 s    | 36.25         | −254         |
| Spectre conservative | 524 950 steps | 4.2 s    | 36.25         | −204         |
| Spectre moderate     | 422 977 steps | 3.2 s    | 36.21         | −171         |

Since the third harmonic falls three times as fast as the input amplitude, it
meets that floor as soon as the input is reduced, and each simulation follows
the correct curve only as long as it stays above its own floor.

| U [dBFS] | event-driven | conservative | moderate |
| -------- | ------------ | ------------ | -------- |
| −10.46   | 35.82        | 35.82        | 35.81    |
| −20.00   | 36.25        | 36.25        | 36.21    |
| −30.46   | 36.30        | 36.31        | 40.31    |
| −40.00   | 36.31        | 35.33        | 26.20    |
| −50.46   | 36.12        | 17.52        | 8.83     |

The remedy in a circuit simulator is a smaller time step. At an input
amplitude of 0.01, where the conservative settings are 1 dB off, it takes about
a thousand time steps per self-oscillation period to reach the reference
result. Reducing the time step by another decade does not improve it any
further, since the round-off accumulated over so many steps then takes over.

| simulator    | maximum time step | state updates | run time | A_oip3 [dBFS] |
| ------------ | ----------------- | ------------- | -------- | ------------- |
| Spectre      | 1e-2 / f0         | 5.3e5         | 4.3 s    | 35.33         |
| Spectre      | 1e-3 / f0         | 4.0e6         | 30 s     | 36.33         |
| Spectre      | 1e-4 / f0         | 3.9e7         | 287 s    | 36.27         |
| event-driven | —                 | 7.9e3         | 8.0 s    | 36.31         |
