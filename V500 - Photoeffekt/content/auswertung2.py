import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.constants as const

lambada,ug,ug_error			=np.genfromtxt("../Werte/intersept.txt").T

lambada=lambada*10**(-9)
freq= const.c/lambada
#lambada=np.delete(lambada,0)
#ug=np.delete(ug,0)
#ug_error=np.delete(ug_error,0)

def linear(x,m,b):
	return m*x+b

#plt.plot(freq,ug,"*")
plt.errorbar(freq,ug, yerr=ug_error, fmt="+", label="Messwerte")
parameter,unfug=curve_fit(linear,freq,ug,sigma=ug_error)
errors = np.sqrt(np.diag(unfug))
x_plot=np.arange(4.2,10)
x_plot=x_plot*10**14
x_plot2=np.arange(0,5.5)
x_plot2=x_plot2*10**14
plt.xlim(0,9*10**14)
plt.xticks([1*10**14,2*10**14,3*10**14,4*10**14,5*10**14,6*10**14,7*10**14,8*10**14,9*10**14],[1,2,3,4,5,6,7,8,9])
plt.xlabel(r"$\nu /(Hz\cdot10^{14})$")
plt.ylabel(r"$U_\mathrm{G} /V$")
plt.ylim(-1.6,2)
plt.grid()
plt.plot(x_plot,linear(x_plot,*parameter),"k", label="Fit")
plt.plot(x_plot2,linear(x_plot2,*parameter),"k--")
plt.legend(loc="best")
plt.savefig("../Bilder/unu_diag.png")
plt.show()

print("Abschnitt: (%f +/- %f) eV"%(parameter[1],errors[1]))
print("Steigung: (%f +/- %f)*10^(-15) a.u."%(parameter[0]*10**15,errors[0]*10**15))