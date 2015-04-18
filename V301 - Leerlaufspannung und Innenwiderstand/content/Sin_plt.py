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

I_S, U_S = np.genfromtxt('../Werte/Sin.txt').T
### Ausgleichsrechnung

def f(x, a, b):
	return a * x + b
params_S, cov_S = curve_fit(f, I_S, U_S)
errors_S = np.sqrt(np.diag(cov_S))
### Unsicherheit der Messgeräte als Fehlerbalken
I_S_err = I_S * 0.03
U_S_err = U_S * 0.03
plt.errorbar(I_S, U_S, xerr = I_S_err, yerr = U_S_err, fmt = 'r.', label = 'Messdaten')

### Diagramm

x = np.linspace(0.0, 0.0009)
plt.plot(x, f(x, *params_S), 'k-', label = 'Regression')

plt.xticks([0, 0.0002, 0.0004, 0.0006, 0.0008],
           ["0", "0.2", "0.4", "0.6", "0.8"])
plt.ylim(0.0, 0.5)
plt.xlabel(r'$I / \mathrm{mA}$')
plt.ylabel(r'$U_{\mathrm{K}} / \mathrm{V}$')
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig("plot_Sin.pdf")
plt.show()
plt.close()

a = ufloat(params_S[0],errors_S[0]) 
b = ufloat(params_S[1],errors_S[1])

print('Sinusspannung')
print('R=',a,'ohm')
print('U_0=',b,'V')

