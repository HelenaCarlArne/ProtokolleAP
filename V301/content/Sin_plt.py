############### Messung 4:Sinusspannung #########
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.constants as const
import uncertainties
from uncertainties import ufloat
### Daten einlesen

I_R, U_R = np.genfromtxt('../Werte/Sin.txt').T
### Ausgleichsrechnung

def f(x, a, b):
	return a * x + b
params_MZ_R, cov_MZ_R = curve_fit(f, I_R, U_R)

### Unsicherheit der Messgeräte als Fehlerbalken
I_R_err = I_R * 0.03
U_R_err = U_R * 0.03
plt.errorbar(I_R, U_R, xerr = I_R_err, yerr = U_R_err, fmt = 'r.', label = 'Messdaten')

### Diagramm

x = np.linspace(0.0, 0.9)
plt.plot(x, f(x, *params_MZ_R), 'k-', label = 'Regression')


plt.ylim(0.0, 0.5)
plt.xlabel(r'$I / \mathrm{mA}$')
plt.ylabel(r'$U_{\mathrm{K}} / \mathrm{V}$')
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig("build\plot_Sin.pdf")
plt.show()
plt.close()



