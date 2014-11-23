#
# Header
#

# import numpy as np 									                                                    
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# import math
# import uncertainties.unumpy as unp
# from uncertainties.unumpy import nominal_values as noms
# from uncertainties.unumpy import std_devs as stds
# from uncertainties import ufloat
# from scipy.stats import sem

#
# Werteeingabe gemäß Symbole
#

cw=4.18
Ty=91.2
Tx=21.6
Tm=52.0
mx=296
my=270

#
# Berechnung
#

# Formel für Wärmekapazität im Dewargefäß
# Masse in Gramm, c_w auf Gramm genormt!
# Einheit ist J/K

print(((cw*my*(Ty-Tm))-(cw*mx*(Tm-Tx)))/(Tm-Tx))
