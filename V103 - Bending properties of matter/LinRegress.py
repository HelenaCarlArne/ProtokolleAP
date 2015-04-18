import numpy as np 								#### Header
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
from uncertainties import ufloat
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12
def linregress(x, y):
    N = len(y) # Annahme: len(x) == len(y), sonst kommt während der Rechnung eine Fehlermeldung
    Delta = N*np.sum(x**2)-(np.sum(x))**2

    A = (N*np.sum(x*y)-np.sum(x)*np.sum(y))/Delta
    B = (np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / Delta

    sigma_y = np.sqrt(np.sum((y - A * x - B)**2) / (N - 2))

    A_error = sigma_y * np.sqrt(N / Delta)
    B_error = sigma_y * np.sqrt(np.sum(x**2) / Delta)
    print (A, A_error, B, B_error)

xrecht,drecht = np.genfromtxt('RechteckigerStab1.txt').T
xrund,drund = np.genfromtxt('RunderStab1.txt').T
x2,d2links, d2rechts = np.genfromtxt('Zweiseitig.txt').T

x_lin_zweiseitig = 3*(550)**2*x2-4*x2**3
x_lin_rund = 492*xrund**2-xrund**3/3
x_lin_rechteckig =524*xrecht**2-xrecht**3/3

print("Es gelte die Reihenfolge:")
print("Steigung m, Fehler, Achsenabschnitt b, Fehler")
print("")
print("Zweiseitige Einspannung, links")
linregress(x_lin_zweiseitig, d2links)
print("")
print("Zweiseitige Einspannung, rechts")
linregress(x_lin_zweiseitig, d2rechts)	
print("")					##
print("Einseitige Einspannung, rund")  
linregress(x_lin_rund, drund)
print("")
print("Einseitige Einspannung, rechteckig")
linregress(x_lin_rechteckig, drecht)








###