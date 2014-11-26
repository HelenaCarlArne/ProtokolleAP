############### Messung 1: Innenwiderstand R_i der Monozelle #########
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.constants as const
import uncertainties
from uncertainties import ufloat
### Daten einlesen

I_R, U_R = np.genfromtxt('../Werte/MZ_R.txt').T
### Ausgleichsrechnung

def f(x, a, b):
	return a * x + b
params_MZ_R, cov_MZ_R = curve_fit(f, I_R, U_R)

### Unsicherheit der Messgeräte als Fehlerbalken
I_R_err = I_R * 0.03
U_R_err = U_R * 0.03
plt.errorbar(I_R, U_R, xerr = I_R_err, yerr = U_R_err, fmt = 'r.', label = 'Messdaten')

### Diagramm

#x = np.linspace(0, 0.3)
#plt.plot(x, f(x, *params_MZ_R), 'k-', label = 'Regression')


#plt.ylim(0.0, 2)
#plt.xlabel(r'$I / \mathrm{A}$')
#plt.ylabel(r'$U##_{\mathrm{K}} / \mathrm{V}$')
#plt.legend(loc = "best")
#plt.tight_layout()
#plt.savefig("build\plot_MZ_R.pdf")
#plt.show()
#plt.close()








############### Messung 2: Monozelle mit Gegenspannung #########

### Daten einlesen

I_UG, U_UG = np.genfromtxt('../Werte/MZ_UG.txt').T
### Ausgleichsrechnung

def f(x, a, b):
	return a * x + b
params_MZ_UG, cov_MZ_UG = curve_fit(f, I_UG, U_UG)

### Unsicherheit der Messgeräte als Fehlerbalken
I_UG_err = I_UG * 0.03
U_UG_err = U_UG * 0.03
plt.errorbar(I_UG, U_UG, xerr = I_UG_err, yerr = U_UG_err, fmt = 'b.', label = 'Messdaten Gegenspannung')

### Diagramm

x = np.linspace(0, 0.35)
plt.plot(x, f(x, *params_MZ_UG), 'k-', label = 'Regression')

plt.plot(x, f(x, *params_MZ_R), 'k-')

plt.ylim(0.0, 3.5)
plt.xlabel(r'$I / \mathrm{A}$')
plt.ylabel(r'$U##_{\mathrm{K}} / \mathrm{V}$')
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig("build\plot_MZ.pdf")
plt.show()
plt.close()
