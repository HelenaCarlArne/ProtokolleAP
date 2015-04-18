############### Messung 3:Rechteckspannung #########
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.constants as const
import uncertainties
from uncertainties import ufloat
### Daten einlesen

I_R, U_R = np.genfromtxt('../Werte/Recht.txt').T
### Ausgleichsrechnung

def f(x, a, b):
	return a * x + b
params_R, cov_R = curve_fit(f, I_R, U_R)
errors_R = np.sqrt(np.diag(cov_R))
### Unsicherheit der Messgeräte als Fehlerbalken
I_R_err = I_R * 0.03
U_R_err = U_R * 0.03
plt.errorbar(I_R, U_R, xerr = I_R_err, yerr = U_R_err, fmt = 'r.', label = 'Messdaten')
a = ufloat(params_R[0],errors_R[0]) 
b = ufloat(params_R[1],errors_R[1])
### Diagramm

x = np.linspace(0, 0.005)
plt.plot(x, f(x, *params_R), 'k-', label = 'Regression')

plt.xticks([0, 0.001, 0.002, 0.003, 0.004, 0.005],
           ["0", "1", "2", "3", "4", "5"])

plt.ylim(0.0, 0.3)
plt.xlabel(r'$I / \mathrm{mA}$')
plt.ylabel(r'$U_{\mathrm{K}} / \mathrm{V}$')
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig("plot_Recht.pdf")
plt.show()
plt.close()

print('Rechteckspannung')
print('R=',a,'ohm')
print('U_0=',b,'V')


