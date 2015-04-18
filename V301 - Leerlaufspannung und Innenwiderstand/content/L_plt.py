############### Leistung Graph #########
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




plt.errorbar(U_R / I_R, U_R * I_R, xerr = U_R / I_R * np.sqrt(0.03**2 + 0.03**2), yerr = U_R * I_R * np.sqrt(0.03**2 + 0.03**2), fmt = 'r.', label = 'Errechnete Messpunkte')
def L(R_a):
    return (1.56**2 * R_a / (5.32 + R_a)**2)
R_a = np.linspace(0,27,900)
plt.plot(R_a, L(R_a), 'k-', label = 'Theoriekurve')

plt.xlim(0.0, 26)
plt.ylim(0.0, 0.15)
plt.xlabel(r'$R_a/ \mathrm{\Omega}$')
plt.ylabel(r'$N/ \mathrm{W}$')
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig("plot_L.pdf")
plt.show()
plt.close()

