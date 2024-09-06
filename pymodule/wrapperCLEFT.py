import ctypes
import os
from ctypes import pointer
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.ctypeslib import ndpointer
from GS import model_CLEFT as model_CLEFT_py_GS

# This is the same as numpy.ctypeslib.load_library
dir_name = "/home/mkarcher/CLEFT_GS"
CLEFT_library = ctypes.CDLL(f"{dir_name}/libCLEFT.so")

# Create/load the loading function
load_CLEFT_wrapped = CLEFT_library.load_CLEFT_wrappable
# Specify the return and argument types of this function
load_CLEFT_wrapped.restype = ctypes.c_void_p
load_CLEFT_wrapped.argtypes = (
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int,
)

# Create/load the prediction functions
model_ZA_wrapped = CLEFT_library.get_prediction_ZA
model_ZA_wrapped_templatefit = CLEFT_library.get_prediction_ZA_tmp_fitting
model_CLPT_wrapped = CLEFT_library.get_prediction_CLPT
model_CLPT_wrapped_templatefit = CLEFT_library.get_prediction_CLPT_tmp_fitting
model_CLPT_wrapped_only_xi_realspace = (
    CLEFT_library.get_prediction_CLPT_only_xi_realspace
)

model_CLEFT_wrapped = CLEFT_library.get_prediction_CLEFT
model_CLEFT_wrapped_fast = CLEFT_library.get_prediction_CLEFT_approx_fast
model_CLEFT_wrapped_templatefit = CLEFT_library.get_prediction_CLEFT_tmp_fitting
model_CLEFT_wrapped_upto8 = CLEFT_library.get_prediction_CLEFT_upto8

# Create/load the free function (only needed for template fitting)
free_wrapped_templatefit = CLEFT_library.free_CLEFT
free_wrapped_templatefit.restype = ctypes.c_void_p
free_wrapped_templatefit

# ZA
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
    ctypes.c_double,
)

# ZA template fitting
# Specify the return and argument data types
model_ZA_wrapped_templatefit.restype = ctypes.c_void_p
model_ZA_wrapped_templatefit.argtypes = (
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
)

# CLPT
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
    ctypes.c_double,
)

# CLPT template fitting
# Specify the return and argument data types
model_CLPT_wrapped_templatefit.restype = ctypes.c_void_p
model_CLPT_wrapped_templatefit.argtypes = (
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
)

# CLPT real space
# Specify the return and argument data types for the xi-only CLPT model in realspace
model_CLPT_wrapped_only_xi_realspace.restype = ctypes.c_void_p
model_CLPT_wrapped_only_xi_realspace.argtypes = (
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_double,
    ctypes.c_double,
)

# CLEFT cumulant version
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
    ctypes.c_double,
)

# CLEFT cumulant version approximate fast version
# Specify the return and argument data types
model_CLEFT_wrapped_fast.restype = ctypes.c_void_p
model_CLEFT_wrapped_fast.argtypes = (
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
    ctypes.c_double,
)



# CLEFT for template fitting
# Specify the return and argument data types
model_CLEFT_wrapped_templatefit.restype = ctypes.c_void_p
model_CLEFT_wrapped_templatefit.argtypes = (
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
    ctypes.c_double,
)


# CLEFT with multipoles up to 8
# Specify the return and argument data types
model_CLEFT_wrapped_upto8.restype = ctypes.c_void_p
model_CLEFT_wrapped_upto8.argtypes = (
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
    ctypes.c_double,
)


# Wrapper for ZA
# This is a new wrapper where we give directly a pointer to the numpy array of the ingredients
def model_ZA(ingredients, theta, s_array, ns):
    """
    Wrapper for ZA.

    Parameters:
    -----------
    ingredients : numpy.ndarray
        Array of ingredients.
    theta : tuple or list
        Tuple or List containing model parameters.
    s_array : numpy.ndarray
        Array of s-values.
    ns : int
        Number of s-values.

    Returns:
    --------
    numpy.ndarray
        Resulting array.
    """
    load_CLEFT_wrapped(ingredients, len(ingredients[:, 0]))
    f, b1, sigv, alpha_par, alpha_per = theta
    res = np.zeros(3 * ns, dtype=np.double)
    model_ZA_wrapped(s_array, ns, res, f, b1, sigv, alpha_par, alpha_per)
    return res


# Wrapper for CLPT
# This is a new wrapper where we give directly a pointer to the numpy array of the ingredients
def model_CLPT(ingredients, theta, s_array, ns):
    """
    Wrapper for CLPT.

    Parameters:
    -----------
    ingredients : numpy.ndarray
        Array of ingredients.
    theta : tuple or list
        Tuple or List containing model parameters.
    s_array : numpy.ndarray
        Array of s-values.
    ns : int
        Number of s-values.

    Returns:
    --------
    numpy.ndarray
        Resulting array.
    """
    load_CLEFT_wrapped(ingredients, len(ingredients[:, 0]))
    f, b1, b2, sigv, alpha_par, alpha_per = theta
    res = np.zeros(3 * ns, dtype=np.double)
    model_CLPT_wrapped(s_array, ns, res, f, b1, b2, sigv, alpha_par, alpha_per)
    return res


# Wrapper for CLPT xi in real space
# This is a new wrapper where we give directly a pointer to the numpy array of the ingredients
def model_CLPT_only_xi_real(ingredients, theta, s_array, ns):
    """
    Wrapper for CLPT xi in real space.

    Parameters:
    -----------
    ingredients : numpy.ndarray
        Array of ingredients.
    theta : tuple or list
        Tuple or List containing model parameters.
    s_array : numpy.ndarray
        Array of s-values.
    ns : int
        Number of s-values.

    Returns:
    --------
    numpy.ndarray
        Resulting array.
    """
    load_CLEFT_wrapped(ingredients, len(ingredients[:, 0]))
    b1, b2 = theta
    res = np.zeros(ns, dtype=np.double)
    model_CLPT_wrapped_only_xi_realspace(s_array, ns, res, b1, b2)
    return res


# Wrapper for CLEFT using cumulant version for sigma
# This is a new wrapper where we give directly a pointer to the numpy array of the ingredients
def model_CLEFT(ingredients, theta, s_array, ns):
    """
    Wrapper for CLEFT using cumulant version for sigma.

    Parameters:
    -----------
    ingredients : numpy.ndarray
        Array of ingredients.
    theta : tuple or list
        Tuple or List containing model parameters.
    s_array : numpy.ndarray
        Array of s-values.
    ns : int
        Number of s-values.

    Returns:
    --------
    numpy.ndarray
        Resulting array.
    """
    load_CLEFT_wrapped(ingredients, len(ingredients[:, 0]))
    f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per = theta
    res = np.zeros(3 * ns, dtype=np.double)
    model_CLEFT_wrapped(
        s_array, ns, res, f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per
    )
    return res

# Wrapper for CLEFT using cumulant version for sigma fast approximate version
# This is a new wrapper where we give directly a pointer to the numpy array of the ingredients
def model_CLEFT_fast(ingredients, theta, s_array, ns):
    """
    Wrapper for CLEFT using cumulant version for sigma.

    Parameters:
    -----------
    ingredients : numpy.ndarray
        Array of ingredients.
    theta : tuple or list
        Tuple or List containing model parameters.
    s_array : numpy.ndarray
        Array of s-values.
    ns : int
        Number of s-values.

    Returns:
    --------
    numpy.ndarray
        Resulting array.
    """
    load_CLEFT_wrapped(ingredients, len(ingredients[:, 0]))
    f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per = theta
    res = np.zeros(3 * ns, dtype=np.double)
    model_CLEFT_wrapped_fast(
        s_array, ns, res, f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per
    )
    return res




def load_templatefit(ingredients):
    """
    Wrapper function to load ingredients for template fitting.

    Parameters:
    -----------
    ingredients : numpy.ndarray
        Array of ingredients.
    """
    load_CLEFT_wrapped(
        ingredients, len(ingredients[:, 0])
    )  # We can use here load_CLEFT_wrapped as all the ingredient files have the same structure just some columns might have been set to zero (CLPT or ZA)


def model_ZA_templatefit(theta, s_array, ns):
    """
    Wrapper for ZA in template-fitting mode.

    Parameters:
    -----------
    theta : tuple or list
        Tuple or List containing model parameters.
    s_array : numpy.ndarray
        Array of s-values.
    ns : int
        Number of s-values.

    Returns:
    --------
    numpy.ndarray
        Resulting array.
    """
    f, b1, sigv, alpha_par, alpha_per = theta
    res = np.zeros(3 * ns, dtype=np.double)
    model_ZA_wrapped_templatefit(s_array, ns, res, f, b1, sigv, alpha_par, alpha_per)
    return res


def model_CLPT_templatefit(theta, s_array, ns):
    """
    Wrapper for CLPT in template-fitting mode.

    Parameters:
    -----------
    theta : tuple or list
        Tuple or List containing model parameters.
    s_array : numpy.ndarray
        Array of s-values.
    ns : int
        Number of s-values.

    Returns:
    --------
    numpy.ndarray
        Resulting array.
    """
    f, b1, b2, sigv, alpha_par, alpha_per = theta
    res = np.zeros(3 * ns, dtype=np.double)
    model_CLPT_wrapped_templatefit(
        s_array, ns, res, f, b1, b2, sigv, alpha_par, alpha_per
    )
    return res


def model_CLEFT_templatefit(theta, s_array, ns):
    """
    Wrapper for CLEFT in template-fitting mode.

    Parameters:
    -----------
    theta : tuple or list
        Tuple or List containing model parameters.
    s_array : numpy.ndarray
        Array of s-values.
    ns : int
        Number of s-values.

    Returns:
    --------
    numpy.ndarray
        Resulting array.
    """
    f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per = theta
    res = np.zeros(3 * ns, dtype=np.double)
    model_CLEFT_wrapped_templatefit(
        s_array, ns, res, f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per
    )
    return res


def free_templatefit():
    free_wrapped_templatefit()


def model_CLEFT_upto8(ingredients, theta, s_array, ns):
    """
    Wrapper for CLEFT computing multipoles up to l=8.

    Parameters:
    -----------
    ingredients : numpy.ndarray
        Array of ingredients.
    theta : tuple or list
        Tuple or List containing model parameters.
    s_array : numpy.ndarray
        Array of s-values.
    ns : int
        Number of s-values.

    Returns:
    --------
    numpy.ndarray
        Resulting array.
    """
    load_CLEFT_wrapped(ingredients, len(ingredients[:, 0]))
    f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per = theta
    res = np.zeros(5 * ns, dtype=np.double)
    model_CLEFT_wrapped_upto8(
        s_array, ns, res, f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per
    )
    return res


# This is for sanity check only
# This is the reference wrapper where we give a filename and load the data from there
def model_load_reference(filename, theta, s_array, ns):
    """
    This is for sanity check only.

    Parameters:
    -----------
    filename : str
        Name of the file to load data from.
    theta : tuple or list
        Tuple or List containing model parameters.
    s_array : numpy.ndarray
        Array of s-values.
    ns : int
        Number of s-values.

    Returns:
    --------
    numpy.ndarray
        Resulting array.
    """
    binput = filename.encode("utf-8")
    init(binput)
    f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per = theta
    res = np.zeros(3 * ns, dtype=np.double)
    model_CLEFT_wrapped(
        s_array, ns, res, f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per
    )
    return res


def main():
    power_spectrum_file = f"{os.path.dirname(__file__)}/../data/power_spectrum.dat"
    test_ZA = np.loadtxt(f"{power_spectrum_file}.za")
    test_CLPT = np.loadtxt(f"{power_spectrum_file}.clpt")
    test_CLEFT = np.loadtxt(f"{power_spectrum_file}.cleft")

    smin = 22.5
    smax = 197.5
    ns = 36
    s_bins = np.linspace(smin, smax, ns)
    # Some more or less reasonable parameter (just for testing if both loading methods give the same result)
    theta_test_ZA = [0.8, 1.5, 2.5, 1, 1]
    theta_test_CLPT = [0.8, 1.5, 0.1, 2.5, 1, 1]
    theta_test_CLEFT = [0.8, 1.5, 0.1, 0.1, 40, 30, 30, 1, 1]

    # ZA model
    result_ZA = model_ZA(test_ZA, theta_test_ZA, s_bins, ns)
    # CLPT model
    result_CLPT = model_CLPT(test_CLPT, theta_test_CLPT, s_bins, ns)
    # CLEFT model with Gaussian streaming in C
    start = time.perf_counter()
    result_CLEFT = model_CLEFT(test_CLEFT, theta_test_CLEFT, s_bins, ns)
    end_C = time.perf_counter()
    # CLEFT model with Gaussian streaming in python 
    result_CLEFT_pyGS = model_CLEFT_py_GS(test_CLEFT, theta_test_CLEFT, s_bins, ns)
    end_py = time.perf_counter()
    print(f'Time needed for GS in C: {end_C-start} seconds')
    print(f'Time needed for GS in Python: {end_py-end_C} seconds')

    start_fast = time.perf_counter()
    result_CLEFT_fast = model_CLEFT_fast(test_CLEFT, theta_test_CLEFT, s_bins, ns)
    end_fast = time.perf_counter()
    print(f'Time needed for fast GS: {end_fast-start_fast} seconds')

    # Test the template fitting
    load_templatefit(test_ZA)
    result_ZA_tmp = model_ZA_templatefit(theta_test_ZA, s_bins, ns)
    free_templatefit()

    load_templatefit(test_CLPT)
    result_CLPT_tmp = model_CLPT_templatefit(theta_test_CLPT, s_bins, ns)
    free_templatefit()

    load_templatefit(test_CLEFT)
    result_CLEFT_tmp = model_CLEFT_templatefit(theta_test_CLEFT, s_bins, ns)
    free_templatefit()

    result_reference = model_load_reference(
        power_spectrum_file, theta_test_CLEFT, s_bins, ns
    )

    # This should be zero
    print(
        f"Sanity check for new load function: {np.allclose(result_CLEFT, result_reference)}"
    )
    print(
        f"Sanity check for template fitting (ZA) :{np.allclose(result_ZA_tmp, result_ZA)}"
    )
    print(
        f"Sanity check for template fitting (CLPT) :{np.allclose(result_CLPT_tmp, result_CLPT)}"
    )
    print(
        f"Sanity check for template fitting (CLEFT) :{np.allclose(result_CLEFT_tmp, result_CLEFT)}"
    )
    print(
        f"Sanity check python streaming: {np.allclose(result_CLEFT, result_CLEFT_pyGS)}"
    )
    print(
        f"Sanity check fast CLEFT: {np.allclose(result_CLEFT, result_CLEFT_fast)}"
    )
    print("Relative differences between python- and C-version for first three multipoles:")
    print(result_CLEFT[:6]/result_CLEFT_pyGS[:6]-1)
    print(result_CLEFT[ns:ns+6]/result_CLEFT_pyGS[ns:ns+6]-1)
    print(result_CLEFT[2*ns:2*ns+6]/result_CLEFT_pyGS[2*ns:2*ns+6]-1)

    print("Relative differences between fast and slow version for first three multipoles:")
    print(result_CLEFT[:6]/result_CLEFT_fast[:6]-1)
    print(result_CLEFT[ns:ns+6]/result_CLEFT_fast[ns:ns+6]-1)
    print(result_CLEFT[2*ns:2*ns+6]/result_CLEFT_fast[2*ns:2*ns+6]-1)

    #############################################################
    # Plotting
    #############################################################
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(13, 4)

    # plot ZA
    ax[0].plot(s_bins, s_bins**2 * result_ZA[0:ns], label="ZA")
    ax[1].plot(s_bins, s_bins**2 * result_ZA[ns : 2 * ns])
    ax[2].plot(s_bins, s_bins**2 * result_ZA[2 * ns :])

    # plot CLPT
    ax[0].plot(s_bins, s_bins**2 * result_CLPT[0:ns], label="CLPT", linestyle="--")
    ax[1].plot(s_bins, s_bins**2 * result_CLPT[ns : 2 * ns], linestyle="--")
    ax[2].plot(s_bins, s_bins**2 * result_CLPT[2 * ns :], linestyle="--")

    # plot CLEFT
    ax[0].plot(s_bins, s_bins**2 * result_CLEFT[0:ns], label="CLEFT")
    ax[1].plot(s_bins, s_bins**2 * result_CLEFT[ns : 2 * ns])
    ax[2].plot(s_bins, s_bins**2 * result_CLEFT[2 * ns :])
    ax[0].plot(
        s_bins,
        s_bins**2 * result_reference[0:ns],
        linestyle="--",
        label="CLEFT sanity check",
    )
    ax[1].plot(s_bins, s_bins**2 * result_reference[ns : 2 * ns], linestyle="--")
    ax[2].plot(s_bins, s_bins**2 * result_reference[2 * ns :], linestyle="--")

    ax[0].plot(s_bins, s_bins**2 * result_CLEFT_pyGS[0:ns], label="CLEFT GS python")
    ax[1].plot(s_bins, s_bins**2 * result_CLEFT_pyGS[ns : 2 * ns])
    ax[2].plot(s_bins, s_bins**2 * result_CLEFT_pyGS[2 * ns :])

    ax[0].set_ylabel(r"Monopole $\xi_0$")
    ax[1].set_ylabel(r"Quadrupole $\xi_2$")
    ax[2].set_ylabel(r"Hexdecapole $\xi_4$")

    for i in range(3):
        ax[i].set_xlabel("s [Mpc/h]")

    ax[0].legend()
    plt.savefig("Test_2PCF.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # This is just for the sanity check
    init = CLEFT_library.load_CLEFT
    init.restype = ctypes.c_void_p
    init.argtypes = [ctypes.c_char_p]

    main()
