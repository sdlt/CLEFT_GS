from pyCLEFT import ffi, lib
import numpy as np
import matplotlib.pyplot as plt

class CLEFT:
    def __init__(self, input, smin, smax, nbins):
        self.input = input.encode("utf-8")
        self.smin = smin
        self.smax = smax
        self.nbins = nbins
        lib.load_CLEFT(self.input)

    def free(self):
        lib.free_CLEFT()

    def predict(self,f,b1,b2,bs,ax,av,aas,apar,aper):
        d = np.zeros(3*self.nbins)
        pd = ffi.cast("double *", d.ctypes.data)
        lib.get_prediction_CLEFT(self.smin, self.smax, self.nbins, pd, f, b1, b2, bs, ax, av, aas, apar, aper)
        return d

    def predict_orders(self,f,b1,b2,bs,ax,av,aas,apar,aper):
        d = np.zeros(3*self.nbins)
        pd = ffi.cast("double *", d.ctypes.data)
        lib.get_prediction_CLEFT(self.smin, self.smax, self.nbins, pd, f, b1, b2, bs, ax, av, aas, apar, aper)
        return d[0:self.nbins], d[self.nbins:2*self.nbins], d[2*self.nbins:]
    
    def scales(self):
        hds = (self.smax-self.smin)/float(self.nbins)*0.5
        s = np.linspace(self.smin+hds,self.smax-hds,self.nbins)
        return s


#--- Example usage ---#
    
pkfile = 'pk_test_z0.9'
model = CLEFT(pkfile, 0., 200., 40)

s = model.scales()
x0,x2,x4 = model.predict_orders(0.5, 0.3, -0.5, 0.1, 0.1, 5, 1.0, 1.0, 1.0)
model.free()

plt.figure()
plt.plot(s,s**2*x0,'-')
plt.plot(s,s**2*x2,'-')
plt.plot(s,s**2*x4,'-')
plt.show()
