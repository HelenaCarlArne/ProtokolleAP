import numpy as np 								#### Header
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from uncertainties import ufloat
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12


d = np.genfromtxt('Durchmesser.txt').T 			#### Einladen der Daten
s1 = np.genfromtxt('Seitex.txt').T
s2 = np.genfromtxt('Seitey.txt').T
xrecht,drecht = np.genfromtxt('RechteckigerStab1.txt').T
xrund,drund = np.genfromtxt('RunderStab1.txt').T
x2,d2links, d2rechts = np.genfromtxt('Zweiseitig.txt').T

#def schoenePlots(y):							#### Funktion, um x-Array zu drehen
#	return 550-y
#x3 = schoenePlots(x2)
#x3 = x3[::-1]
#d2rechts = d2rechts[::-1]

#print(x2)
#print(x3)

########## Der erste Plot ##############

plt.plot(xrecht,drecht,label='Rechteckiger Stab') 
plt.plot(xrund,drund,label="Runder Stab")
plt.plot(x2,d2rechts,label="Zweiseitige Einspannung, rechts")
plt.plot(x2,d2links,label="Zweiseitige Einspannung, links")
plt.legend(loc="best")
plt.xlabel("x in mm")
plt.ylabel("Auslenkung in mm")
plt.title("Auslenkungen direkt")
plt.show()

############# Der zweite und dritte Plot ###############

def f(x, m, b):
    return m*x+b
t = np.linspace(0,70000000)
u = np.linspace(30000000,180000000)
#### Einseitig
popt, pcov = curve_fit(f, 492*xrund**2-xrund**3/3, drund)
plt.plot(t, f(t, *popt), 'r-', label='Fit')
print("Runder Stab: m={} und b={} ".format(*popt))

popt, pcov = curve_fit(f, 524*xrecht**2-xrecht**3/3, drecht)
plt.plot(t, f(t, *popt), 'b-', label='Fit')
print("Rechteckiger Stab: m={} und b={} ".format(*popt))
print("")

plt.plot(492*xrund**2-xrund**3/3,drund, "rx", label="Rund")
plt.plot(524*xrecht**2-xrecht**3/3,drecht, "bx",label="Rechteckig")

plt.legend(loc="best")
plt.title("Einseitige Einspannung, linearisiert")
plt.xlabel(r'$492x^2 - \frac{1}{3}x^3$')
plt.ylabel("Auslenkung in mm")
plt.show()

#### Zweiseitig
plt.plot(3*(550)**2*x2-4*x2**3,d2rechts, "rx", label="Rechts")
plt.plot(3*(550)**2*x2-4*x2**3,d2links, "bx", label="Links")

popt, pcov = curve_fit(f, 3*(550)**2*x2-4*x2**3,d2rechts)
plt.plot(u, f(u, *popt), 'r-', label='Fit')
print("Zweiseitige Einspannung 1: m={} und b={} ".format(*popt))

popt, pcov = curve_fit(f, 3*(550)**2*x2-4*x2**3, d2links)
plt.plot(u, f(u, *popt), 'b-', label='Fit')
print("Zweiseitige Einspannung 2: m={} und b={} ".format(*popt))
print("")

plt.legend(loc="best")
plt.title("Zweiseitige Einspannung, linearisiert")
plt.xlabel(r"$3(550)^2x-4x^3$")
plt.ylabel("Auslenkung in mm")
plt.show()
