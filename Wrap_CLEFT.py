import ctypes
import numpy as np
from ctypes import pointer
from numpy.ctypeslib import ndpointer
import matplotlib.pyplot as plt

# This is the same as numpy.ctypeslib.load_library
CLEFT_library = ctypes.CDLL("/home/mkarcher/CLEFT_GS/libCLEFT.so")

# Create/load the loading function
load_CLEFT_wrapped = CLEFT_library.load_CLEFT_wrappable
# Specify the return and argument types of this function
load_CLEFT_wrapped.restype = ctypes.c_void_p
load_CLEFT_wrapped.argtypes = (ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'), ctypes.c_int)

# Create/load the free and the prediction function
free_wrapped = CLEFT_library.free_CLEFT
model_wrapped = CLEFT_library.get_prediction_CLEFT

# Specify the return and argument data types
model_wrapped.restype = ctypes.c_void_p
model_wrapped.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_int, ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)

# This is a new wrapper where we give directly a pointer to the numpy array of the ingredients
def model_CLEFT(ingredients, theta, smin, smax, ns):
    load_CLEFT_wrapped(ingredients, len(ingredients[:,0]))
    f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per = theta
    res = np.zeros(3*ns, dtype=np.double)
    model_wrapped(smin, smax, ns, res, f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per)
    free_wrapped()
    return res

# This is the vanilla wrapper where we give a filename and load the data from there
def model_load_vanilla(filename, theta, smin, smax, ns):
    binput = filename.encode('utf-8')
    init(binput)
    f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per = theta
    res = np.zeros(3*ns, dtype=np.double)
    model_wrapped(smin, smax, ns, res, f, b1, b2, bs, ax, av, aas, alpha_par, alpha_per)
    free_wrapped()
    return res



def main():
    test_CLEFT = np.loadtxt("Testing_Plin.dat.cleft")

    # Some more or less reasonable parameter (just for testing if both loading methods give the same
    theta_test = [0.8, 1.5, 0.1, 0.1, 4, 3, 3, 1, 1]
    result = model_CLEFT(test_CLEFT, theta_test, 22.5, 197.5, 36)

    result_vanilla = model_load_vanilla("Testing_Plin.dat", theta_test, 22.5, 197.5, 36)

    # This should be zero
    print(result-result_vanilla)
    r_bins = np.linspace(22.5, 197.5, 36)

    plt.plot(r_bins, r_bins**2*result[0:36])
    plt.plot(r_bins, r_bins**2*result[36:72])
    plt.plot(r_bins, r_bins**2*result[72:108])
    plt.plot(r_bins, r_bins**2*result_vanilla[0:36], linestyle='--')
    plt.plot(r_bins, r_bins**2*result_vanilla[36:72], linestyle='--')
    plt.plot(r_bins, r_bins**2*result_vanilla[72:108], linestyle='--')
    plt.show()


if __name__ == "__main__":
    # This is just for testing purposes so we don't want this to be created if we just load this as a module
    init = CLEFT_library.load_CLEFT
    init.restype = ctypes.c_void_p
    init.argtypes = [ctypes.c_char_p]

    main()




