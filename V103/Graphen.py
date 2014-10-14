import numpy as np 								#### Header
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from uncertainties import ufloat
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12


d = np.genfromtxt('Durchmesser.txt').T 			#### Einladen der Daten
s1 = np.genfromtxt('Seite1.txt').T
s2 = np.genfromtxt('Seite2.txt').T
xrecht,drecht = np.genfromtxt('RechteckigerStab1.txt').T
xrund,drund = np.genfromtxt('RunderStab1.txt').T
x2,d2links, d2rechts = np.genfromtxt('Zweiseitig.txt').T

def schoenePlots(y):							#### Funktion, um x-Array zu drehen
	return 550-y
x3 = schoenePlots(x2)
x3 = x3[::-1]
d2rechts = d2rechts[::-1]

#print(x2)
#print(x3)

plt.plot(xrecht,drecht,label='Rechteckiger Stab')#### Die Plots
plt.plot(xrund,drund,label="Runder Stab")
plt.plot(x3,d2rechts,label="Zweiseitige Einspannung, rechts")
plt.plot(x2,d2links,label="Zweiseitige Einspannung, links")
plt.legend(loc="best")
plt.xlabel("x in mm")
plt.ylabel("Auslenkung in mm")
plt.show()
