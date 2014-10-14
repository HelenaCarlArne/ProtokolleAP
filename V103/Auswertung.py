
import numpy as np 										#### Header
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from uncertainties import ufloat
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 16


d = np.genfromtxt('Durchmesser.txt').T 					#### Einladen der Daten
s1 = np.genfromtxt('Seite1.txt').T
s2 = np.genfromtxt('Seite2.txt').T
xrecht,drecht = np.genfromtxt('RechteckigerStab1.txt').T
xrund,drund = np.genfromtxt('RunderStab1.txt').T
x2,d2rechts, d2links = np.genfromtxt('Zweiseitig.txt').T

durch = ufloat(np.mean(d),np.std(d)) 					#### Berechnung der Werte
seite1 = ufloat(np.mean(s1),np.std(s1))
seite2 = ufloat(np.mean(s2),np.std(s2))


print("Der Durchmesser betraegt:",durch,"mm") 			#### Ausgabe
print("Die eine Seite betraegt:",seite1,"mm")
print("Der Durchmesser betraegt:",seite2,"mm")