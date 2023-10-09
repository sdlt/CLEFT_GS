import ctypes
import os
from ctypes import pointer

import matplotlib.pyplot as plt
import numpy as np
from numpy.ctypeslib import ndpointer

# This is the same as numpy.ctypeslib.load_library
# CLEFT_library = ctypes.CDLL("/home/mkarcher/CLEFT_GS/libCLEFT.so")
dir_name = "/home/mkarcher/CLEFT_GS/"
CLEFT_library = ctypes.CDLL(f"{dir_name}/libCLEFT.so")

# Create/load the loading function
load_CLEFT_wrapped = CLEFT_library.load_CLEFT_wrappable
# Specify the return and argument types of this function
load_CLEFT_wrapped.restype = ctypes.c_void_p
load_CLEFT_wrapped.argtypes = (
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int,
)

# Create/load the free and the prediction function
model_ZA_wrapped = CLEFT_library.get_prediction_ZA
model_CLPT_wrapped = CLEFT_library.get_prediction_CLPT
model_CLPT_wrapped_only_xi_realspace = CLEFT_library.get_prediction_CLPT_only_xi_realspace
model_CLEFT_wrapped = CLEFT_library.get_prediction_CLEFT

# Specify the return and argument data types
model_ZA_wrapped.restype = ctypes.c_void_p
model_ZA_wrapped.argtypes = (
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double
)

# Specify the return and argument data types
model_CLPT_wrapped.restype = ctypes.c_void_p
model_CLPT_wrapped.argtypes = (
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double
)

# Specify the return and argument data types for the xi-only CLPT model in realspace
model_CLPT_wrapped_only_xi_realspace.restype = ctypes.c_void_p
model_CLPT_wrapped_only_xi_realspace.argtypes = (
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
)



# Specify the return and argument data types
model_CLEFT_wrapped.restype = ctypes.c_void_p
model_CLEFT_wrapped.argtypes = (
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double
)

# This is a new wrapper where we give directly a pointer to the numpy array of the ingredients
def model_ZA(ingredients, theta, s_array, ns):
    load_CLEFT_wrapped(ingredients, len(ingredients[:, 0]))
    f, b1, sigv, alpha_par, alpha_per = theta
    res = np.zeros(3 * ns, dtype=np.double)
    model_ZA_wrapped(s_array, ns, res, f, b1, sigv, alpha_par, alpha_per)
    return res

# This is a new wrapper where we give directly a pointer to the numpy array of the ingredients
def model_CLPT(ingredients, theta, s_array, ns):
    load_CLEFT_wrapped(ingredients, len(ingredients[:, 0]))
    f, b1, b2, sigv, alpha_par, alpha_per = theta
    res = np.zeros(3 * ns, dtype=np.double)
    model_CLPT_wrapped(s_array, ns, res, f, b1, b2, sigv, alpha_par, alpha_per)
    return res

# This is a new wrapper where we give directly a pointer to the numpy array of the ingredients
def model_CLPT_only_xi_real(ingredients, theta, s_array, ns):
    load_CLEFT_wrapped(ingredients, len(ingredients[:, 0]))
    b1, b2, bs2, bn2 = theta
    res = np.zeros(ns, dtype=np.double)
    model_CLPT_wrapped_only_xi_realspace(s_array, ns, res, b1, b2, bs2, bn2)
    return res

# This is a new wrapper where we give directly a pointer to the numpy array of the ingredients
def model_CLEFT(ingredients, theta, s_array, ns):
    load_CLEFT_wrapped(ingredients, len(ingredients[:, 0]))
    f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per = theta
    res = np.zeros(3 * ns, dtype=np.double)
    model_CLEFT_wrapped(s_array, ns, res, f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per)
    return res



# This is the vanilla wrapper where we give a filename and load the data from there
def model_load_vanilla(filename, theta, s_array, ns):
    binput = filename.encode("utf-8")
    init(binput)
    f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per = theta
    res = np.zeros(3 * ns, dtype=np.double)
    model_CLEFT_wrapped(s_array, ns, res, f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per)
    return res


def main():
    test_ZA = np.loadtxt("Testing_Plin.dat.za")
    test_CLPT = np.loadtxt("Testing_Plin.dat.clpt")
    test_CLEFT = np.loadtxt("Testing_Plin.dat.cleft")

    smin = 22.5
    smax = 197.5
    ns = 36
    r_bins = np.linspace(smin, smax, ns)
    # Some more or less reasonable parameter (just for testing if both loading methods give the same
    theta_test_ZA = [0.8, 1.5, 2.5, 1, 1]
    theta_test_CLPT = [0.8, 1.5, 0.1, 2.5, 1, 1]
    theta_test_CLEFT = [0.8, 1.5, 0.1, 0.1, 40, 30, 30, 1, 1]

    result_ZA = model_ZA(test_CLEFT, theta_test_ZA, r_bins, ns)
    result_CLPT = model_CLPT(test_CLEFT, theta_test_CLPT, r_bins, ns)
    result_CLEFT = model_CLEFT(test_CLEFT, theta_test_CLEFT, r_bins, ns)

    result_vanilla = model_load_vanilla("Testing_Plin.dat", theta_test_CLEFT, r_bins, ns)

    # This should be zero
    print(f"{np.allclose(result_CLEFT, result_vanilla)=}")

    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(13,4)

    # plot ZA
    ax[0].plot(r_bins, r_bins**2 * result_ZA[0:ns], label='ZA')
    ax[1].plot(r_bins, r_bins**2 * result_ZA[ns : 2 * ns])
    ax[2].plot(r_bins, r_bins**2 * result_ZA[2 * ns :])

    # plot CLPT
    ax[0].plot(r_bins, r_bins**2 * result_CLPT[0:ns], label='CLPT')
    ax[1].plot(r_bins, r_bins**2 * result_CLPT[ns : 2 * ns])
    ax[2].plot(r_bins, r_bins**2 * result_CLPT[2 * ns :])

    # plot CLEFT
    ax[0].plot(r_bins, r_bins**2 * result_CLEFT[0:ns], label='CLEFT')
    ax[1].plot(r_bins, r_bins**2 * result_CLEFT[ns : 2 * ns])
    ax[2].plot(r_bins, r_bins**2 * result_CLEFT[2 * ns :])
    ax[0].plot(r_bins, r_bins**2 * result_vanilla[0:ns], linestyle="--")
    ax[1].plot(r_bins, r_bins**2 * result_vanilla[ns : 2 * ns], linestyle="--")
    ax[2].plot(r_bins, r_bins**2 * result_vanilla[2 * ns :], linestyle="--")


    ax[0].legend()
    plt.savefig("Test_2PCF.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # This is just for testing purposes so we don't want this to be created if we just load this as a module
    init = CLEFT_library.load_CLEFT
    init.restype = ctypes.c_void_p
    init.argtypes = [ctypes.c_char_p]

    main()
