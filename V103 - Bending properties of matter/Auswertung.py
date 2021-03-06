
import numpy as np 										#### Header
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from uncertainties import ufloat
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 16


d = np.genfromtxt('Durchmesser.txt').T 					#### Einladen der Daten
s1 = np.genfromtxt('Seitex.txt').T
s2 = np.genfromtxt('Seitey.txt').T
xrecht,drecht = np.genfromtxt('RechteckigerStab1.txt').T
xrund,drund = np.genfromtxt('RunderStab1.txt').T
x2,d2links, d2rechts = np.genfromtxt('Zweiseitig.txt').T

durch = ufloat(np.mean(d),np.std(d)) 					#### Berechnung der Werte
seite1 = ufloat(np.mean(s1),np.std(s1))
seite2 = ufloat(np.mean(s2),np.std(s2))

Irund = 0.25*(durch/2)**(4)*np.pi					#### Berechnung der Flächenträgheitsmomente
Irecht = 1/12*seite2**3*seite1

print("Der Durchmesser betraegt:",durch,"mm") 			#### Ausgabe
print("Der Radius betraegt:",durch/2,"mm")
print("Die x-Seite betraegt:",seite1,"mm")
print("Der y-Seite betraegt:",seite2,"mm")
print("")
print("Fuer die Flaechentraegheitsmomente gelten:")
print("Rund:",Irund,"mm^4")
print("Recht:",Irecht,"mm^4")
print("")


