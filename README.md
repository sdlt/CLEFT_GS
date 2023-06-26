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

The model can be used as a python module calling the `libCLEFT` library. The CLEFT
python module needs first to be built with `cffi`. For this:

> cd pymodule

> python build.py

The wrapper class is in `wrapperCLEFT.py` and can be tested by running:

> python wrapperCLEFT.py

For the python wrapper to work in other python scripts, `libCLEFT.so` and
`pyCLEFT.cpython-*.so` files need to be put in the same directory.

## Contact

For any question, please send an email to sylvain.delatorre@lam.fr.
