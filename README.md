# CLEFT_GS

This C code implements the CLEFT Gaussian Streaming model. It is based
on CLPT_GS code by Antoine Rocher & Michel-Andrès Breton:
https://github.com/sdlt/CLPT_GS. The final products are the monopole,
quadrupole, and hexadecapole moments of the redshift-space two-point
correlation function. A python wrapper is also provided.

## Requirements

The `gsl`, `openmp` libraries should to be installed. For the python
wrapper, `cffi` and `numpy` modules should be installed.

## Compilation

Compilation is done with

> make all

Individual codes are compiled as:

> make gs

> make cleft

> make libgs

The `GS` and `CLEFT` executables can be ran from the command line.
`CLEFT` computes the ingredients of the CLEFT-GS model,
`GS` computes the prediction of the model for the monopole, quadrupole,
and hexadecapole of the correlation function, and `libCLEFT.so` is a library that
can be used externally in other C codes or wrappers.

## Python wrapper
### Full-shape fitting
The code `Wrap_CLEFT.py` provides wrapper for the ZA, CLPT and CLEFT model to be used for full-shape fits. In case template fitting those functions are not optimal as they free the ingredients after computing the multipoles once. The wrappers to be used for full-shape fitting are

> `model_ZA` fit-params: [f, b_1, s_v, alpha_par, alpha_perp]<br>

> `model_CLPT` fit-params: [f, b_1, b_2, s_v, alpha_par, alpha_perp]<br>

> `model_CLEFT` fit-params: [f, b_1, b_2, b_s, a_xi, a_v, a_sigma, alpha_par, alpha_perp]<br>

They all take an ingredients file first, then an array of fit-parameters (see above), followed by the array of bins in *s* and the number of bins *ns*.
### Template fitting
In template fitting the ingredients stay the same for each MCMC step and only bias and nuisance parameters are varied as well as the growth rate f and AP parameters containing cosmological information. For this case, you can use the dedicated wrapper ending with `_templatefit`. Before the MCMC chain starts you need to initialize the code with an ingredients file once using `load_templatefit`. Inside the MCMC chain you can use `model_*_templatefit` and after the chain has ended you need to call `free_templatefit`.

## Contact
For any question, please send an email to sylvain.delatorre@lam.fr, breton@ice.csic.es or martin.karcher@lam.fr
