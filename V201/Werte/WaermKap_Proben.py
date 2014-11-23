"""

 Information


Dieses Skript gibt die Wärmekapazitäten und die Molwärmen der Proben aus.

In Block "Werteeingabe gemäß Symbole" werden die in der Anleitung gegebenen Werte eingetragen.
In Block "Berechnung" werden Funktionen definiert, die die Wärmekapazitäten ausgibt [warmth()] oder 
die Molwärmen bei konstantem Volumen  [otherwarmth()] ausgibt.
Die Parameter sind so benannt, wie man sie schreiben würde (siehe Funktionendefinition)
In dem Ausgabe-Block sind die Funktionen enthalten, die Parameter stammen vom Experiment/Anleitung.

Die vierte, abgetrennte Zahl bei Blei ist der Mittelwert; 
der zusätzliche Term bei der Molwärmen bei konstantem Volumen ist die Differenz der Molwärmen.
"""

#
# Header
#

import numpy as np 									                                                    
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
import math
import uncertainties.unumpy as unp
# from uncertainties.unumpy import nominal_values as noms
# from uncertainties.unumpy import std_devs as stds
from uncertainties import ufloat
from scipy.stats import sem

#
# Werteeingabe gemäß Symbole
#

cgmg=218.02000000000015
cw=4.18
Mpb=207.2
Mcu=63.5
Mc=120

#
# Berechnung
#

# Formel für Wärmekapazität im Dewargefäß
# Masse in Gramm, c_w auf Gramm genormt!
# Einheit ist J/K

def warmth(mw,mk,Tw,Tk,Tm):
	#print(((cw*mw+cgmg)*(Tm-Tw))/(mk*(Tk-Tm)))
	return(((cw*mw+cgmg)*(Tm-Tw))/(mk*(Tk-Tm)))
def otherwarmth(alpha,kappa,M,rho,T):
	return(9*(alpha**2)*kappa*(M/(rho*10**6))*T)

#
# Datenausgabe
#
print("Spezifische Waermekapazitaeten")
print("Blei")
print(warmth(574,385.55,23.5,82.9,25.9))
print(warmth(537,385.55,21.8,91.0,25.2))
print(warmth(583,385.55,22.3,75.7,24.5))
print("")
# a# sind die Argumente der Funktion warmth. 
# Jedes einzelne Argument wird befehlerrechnet.
a1=ufloat(np.mean([574,537,583]),sem([574,537,583]))
a2=385.55
a3=ufloat(np.mean([23.5,21.8,22.3]),sem([23.5,21.8,22.3]))
a4=ufloat(np.mean([82.9,91.0,75.7]),sem([82.9,91.0,75.7]))
a5=ufloat(np.mean([25.9,25.2,24.5]),sem([25.9,25.2,24.5]))
print(warmth(a1,a2,a3,a4,a5))
print("")
print("Graphit")
print(warmth(544,107.23,22.5,83.3,25.6))
print("")
print("Kupfer")
print(warmth(562,139.77,22.1,82.3,26.4))
print("")
print("")
print("")
print("")

print("Molwaerme bei konstantem Druck")
print("Blei")
print(warmth(574,385.55,23.5,82.9,25.9)*Mpb)
print(warmth(537,385.55,21.8,91.0,25.2)*Mpb)
print(warmth(583,385.55,22.3,75.7,24.5)*Mpb)
print("")
# analog zur Bem. bei Blei oben
a1=ufloat(np.mean([574,537,583]),sem([574,537,583]))
a2=385.55
a3=ufloat(np.mean([23.5,21.8,22.3]),sem([23.5,21.8,22.3]))
a4=ufloat(np.mean([82.9,91.0,75.7]),sem([82.9,91.0,75.7]))
a5=ufloat(np.mean([25.9,25.2,24.5]),sem([25.9,25.2,24.5]))
print(warmth(a1,a2,a3,a4,a5)*Mpb)
print("")
print("Graphit")
print(warmth(544,107.23,22.5,83.3,25.6)*Mc)
print("")
print("Kupfer")
print(warmth(562,139.77,22.1,82.3,26.4)*Mcu)
print("")
print("")
print("")
print("")

print("Molwaerme bei konstantem Volumen")
print("Blei")
pb1=warmth(574,385.55,23.5,82.9,25.9)*Mpb-otherwarmth(29*10**(-6),42*10**9,207.2,11.35,25.9+273.15)
print(pb1)
pb2=warmth(537,385.55,21.8,91.0,25.2)*Mpb-otherwarmth(29*10**(-6),42*10**9,207.2,11.35,25.2+273.15)
print(pb2)
pb3=warmth(583,385.55,22.3,75.7,24.5)*Mpb-otherwarmth(29*10**(-6),42*10**9,207.2,11.35,24.5+273.15)
print(pb3)
print("")
pbm= ufloat(np.mean([pb1,pb2,pb3]),sem([pb1,pb2,pb3]))
print(pbm)
print(otherwarmth(29*10**(-6),42*10**9,207.2,11.35,25.9+273.15))
print("")
print("Graphit")
cm = warmth(544,107.23,22.5,83.3,25.6)*Mc-otherwarmth(8*10**(-6), 33*10**9,12,2.25,25.6+273.15)
print(cm)
print(otherwarmth(8*10**(-6), 33*10**9,12,2.25,25.6+273.15))
print("")
print("Kupfer")
cum = warmth(562,139.77,22.1,82.3,26.4)*Mcu-otherwarmth(16.8*10**(-6),136*10**9,63.5,8.96,26.4+273.15)
print(cum)
print(otherwarmth(16.8*10**(-6),136*10**9,63.5,8.96,26.4+273.15))
print("")
print("")
print("")
print("")
print("Die Abweichungen vom DP-Gesetz")
R = ufloat(8.3144621,0.0000075)
print("Blei")
print((pbm-3*R)/(3*R))
print("")
print("Graphit")
print((cm-3*R)/(3*R))
print("")
print("Kupfer")
print((cum-3*R)/(3*R))