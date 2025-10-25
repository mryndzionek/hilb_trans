# Hilbert transformer filter design resources

This repo gathers code related to Hilbert transformer (FIR) filter design.

## Links

[An Efficient Analytic Signal Generator](http://www.claysturner.com/dsp/ASG.pdf)

[SciPy:Remez](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.remez.html)

[SciPy:firwin2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin2.html)

[Least-Squares Linear-Phase FIR Filter Design](https://ccrma.stanford.edu/~jos/sasp/Least_Squares_Linear_Phase_FIR_Filter.html)

## Plots

![plot_window](plots/plot_window.png)
![plot_parks](plots/plot_parks-mcclellan__remez_.png)
![plot_turner](plots/plot_turner.png)
![plot_least_squares_1](plots/plot_least_squares_pi_2.png)
![plot_least_squares_2](plots/plot_least_squares_pi_4.png)


## Extras

The script `allpass_halfband_des.py` allows for designing an allpass IIR halfband filter
with following structure:

![filter_block](plots/allpass_halfband.png)

Number of sections in each branch is configurable. By default it is two.
Such filter can easily be converted to an allpass Hilbert transformer via the script `des_hilb_allpass.py`.

Halfband allpass IIR filter:

![plot_allpass_halfband](plots/plot_allpass_halfband.png)

Allpass Hilbert transformer:
![plot_allpass_mag](plots/plot_allpass_mag.png)
![plot_allpass_phase](plots/plot_allpass_phase.png)
