import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.stats import sem
from uncertainties import ufloat

alpha, M1, M2, M3, M4, y =np.genfromtxt('../Werte/werte.txt').T

U1=np.genfromtxt('../Werte/1.txt').T
U2=np.genfromtxt('../Werte/2.txt').T
U3=np.genfromtxt('../Werte/3.txt').T
U4=np.genfromtxt('../Werte/4.txt').T
U5=np.genfromtxt('../Werte/5.txt').T
U6=np.genfromtxt('../Werte/6.txt').T
U7=np.genfromtxt('../Werte/7.txt').T
U8=np.genfromtxt('../Werte/8.txt').T
U9=np.genfromtxt('../Werte/9.txt').T

alpha=np.radians(alpha)

U_1=np.mean(U1)
U_1_err=sem(U1)
U_2=np.mean(U2)
U_2_err=sem(U2)
U_3=np.mean(U3)
U_3_err=sem(U3)
U_4=np.mean(U4)
U_4_err=sem(U4)
U_5=np.mean(U5)
U_5_err=sem(U5)
U_6=np.mean(U6)
U_6_err=sem(U6)
U_7=np.mean(U7)
U_7_err=sem(U7)
U_8=np.mean(U8)
U_8_err=sem(U8)
U_9=np.mean(U9)
U_9_err=sem(U9)

print('Mittelwerte und Fehler der Spannungen pro Winkel')
print(U_1,'pm',U_1_err)
print(U_2,'pm',U_2_err)
print(U_3,'pm',U_3_err)
print(U_4,'pm',U_4_err)
print(U_5,'pm',U_5_err)
print(U_6,'pm',U_6_err)
print(U_7,'pm',U_7_err)
print(U_8,'pm',U_8_err)
print(U_9,'pm',U_9_err)


#Maxima liegen bei U4 und U8, Minima bei U2 und U6
#Berechnung des Inklinationswinkels
##HIER FEHLT NOCH DIE FEHLERRECHNUNG
IW1_=U_2/U_4
IW2_=U_6/U_8

IW1=38.11760355
IW2=24.07947132
print('')
print('Inklinationswinkel nach Arnes Formel für geteilte Messung')
print(IW1)
print(IW2)
#print(IW1,'pm',IW1_err)
#print(IW2,'pm',IW2_err)

##PLOTS
#1. Einfach alle Messpunkte aufgetragen gegen den Winkel

plt.plot(alpha,M1,'rx',label='Messung 1')
plt.plot(alpha,M2,'yx',label='Messung 2')
plt.plot(alpha,M3,'gx',label='Messung 3')
plt.plot(alpha,M4,'kx',label='Messung 4')
plt.xlabel('Winkel /°')
plt.ylabel('Spannung U')
plt.legend(loc="best")
plt.savefig("../Bilder/M1-4.pdf")
#plt.show()
plt.close()
#Plot der Mittelwerte aufgetragen gegen den Winkel, hier fehlen Fehler
plt.plot(0,U_1,'rx')
plt.plot(45,U_2,'rx')
plt.plot(90,U_3,'rx')
plt.plot(135,U_4,'rx') 
plt.plot(180,U_5,'rx')
plt.plot(225,U_6,'rx')
plt.plot(270,U_7,'rx')
plt.plot(315,U_8,'rx')
plt.plot(360,U_9,'rx')
plt.xlabel('Winkel /°')
plt.ylabel('Spannung U')
plt.legend(loc="best")
plt.savefig("../Bilder/MW.pdf")
#plt.show()
plt.close()


def f(alpha, a, b, c):
	return a*np.cos(alpha+b)+c
params, cov = curve_fit(f, alpha, y , maxfev=8000000, p0=[1.92172884017,3.16178588224,33.727913358])
errors = np.sqrt(np.diag(cov))

print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('c =', params[2], '±', errors[2])

alpha_plot = np.linspace(0, 7)

plt.plot(alpha, y, 'rx', label="example data")
plt.plot(alpha_plot, f(alpha_plot, *params), 'b-', label='linearer Fit')
plt.legend(loc="best")
#plt.show()
plt.savefig('../Bilder/plot.pdf')
plt.close()

